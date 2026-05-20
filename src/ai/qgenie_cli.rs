// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! AI provider that shells out to the `qgenie` CLI instead of calling an API directly.
//! Uses the local QGenie installation (subscription auth) rather than API credits.
//!
//! ## How it works
//!
//! The prompt is serialised by `build_prompt` (shared with the other CLI providers)
//! and piped to `qgenie agent exec --json` via stdin.  QGenie emits JSONL events on
//! stdout; we collect `item.completed` events (containing `agent_message` items) and
//! return the concatenated text as the `AiResponse`.  As a reliable fallback,
//! `--output-last-message` writes the final agent message to a temp file which is
//! read when the JSONL stream contains no recognised message events.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::env;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

use super::claude_cli::{build_prompt, parse_inner_response};
use crate::ai::{AiProvider, AiRequest, AiResponse, AiUsage, ProviderCapabilities};

pub struct QGenieCliProvider {
    pub model: String,
    pub context_window_size: usize,
}

impl Default for QGenieCliProvider {
    fn default() -> Self {
        Self {
            model: "default".to_string(),
            context_window_size: 200_000,
        }
    }
}

#[async_trait]
impl AiProvider for QGenieCliProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = build_prompt(&request);

        debug!("qgenie-cli prompt length: {} chars", prompt.len());

        // Build args: `qgenie agent exec --json [--output-last-message FILE] [-m MODEL]`
        // The prompt is written to stdin to avoid ARG_MAX limits on large diffs.
        // --output-last-message provides a reliable fallback when the JSONL stream
        // does not contain a recognised agent_message item.
        let mut args = vec![
            "agent".to_string(),
            "exec".to_string(),
            "--json".to_string(),
        ];

        let last_msg_file = tempfile_path();
        args.push("--output-last-message".to_string());
        args.push(last_msg_file.to_string_lossy().to_string());

        if self.model != "default" {
            args.push("-m".to_string());
            args.push(self.model.clone());
        }

        // Pass "-" so qgenie reads the prompt from stdin.
        args.push("-".to_string());

        let mut child = Command::new("qgenie")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                anyhow::anyhow!("Failed to spawn qgenie CLI: {}. Is it installed?", e)
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
            stdin.flush().await?;
        }

        let output = timeout(Duration::from_secs(600), child.wait_with_output())
            .await
            .map_err(|_| anyhow::anyhow!("qgenie CLI timed out after 10 minutes"))?
            .map_err(|e| anyhow::anyhow!("qgenie CLI wait error: {}", e))?;

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                if !line.trim().is_empty() {
                    debug!("[qgenie-cli stderr] {}", line);
                }
            }
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "qgenie CLI exited with {}: {}",
                output.status,
                stderr.trim()
            );
        }

        let raw = String::from_utf8_lossy(&output.stdout);
        let last_msg = std::fs::read_to_string(&last_msg_file).ok();
        let _ = std::fs::remove_file(&last_msg_file);
        parse_qgenie_output(&raw, last_msg.as_deref())
    }

    fn estimate_tokens(&self, request: &AiRequest) -> usize {
        let chars: usize = request
            .messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|c| c.len())
            .sum();
        chars / 4
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: self.model.clone(),
            context_window_size: self.context_window_size,
        }
    }
}

/// Returns a path in the system temp directory for the last-message file.
fn tempfile_path() -> std::path::PathBuf {
    let mut path = env::temp_dir();
    path.push(format!(
        "qgenie-last-msg-{}-{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));
    path
}

/// Parse JSONL output from `qgenie agent exec --json`.
///
/// `last_msg` is the optional content of the `--output-last-message` file written
/// by qgenie, used as a reliable fallback when the JSONL stream contains no
/// recognised `agent_message` items.
///
/// QGenie emits one JSON object per line.  The current event schema is:
///   {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
///   {"type":"turn.completed","usage":{"input_tokens":N,...}}
///
/// Legacy top-level `message` / `agent_message` shapes are also accepted.
fn parse_qgenie_output(raw: &str, last_msg: Option<&str>) -> Result<AiResponse> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut usage: Option<AiUsage> = None;
    debug!("=== QGENIE RAW OUTPUT START ===\n{}\n=== QGENIE RAW OUTPUT END ===", raw);

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };

        match event["type"].as_str() {
            // Current qgenie format: item.completed wraps the agent_message item
            Some("item.completed") => {
                let item = &event["item"];
                match item["type"].as_str() {
                    Some("agent_message") | Some("message") => {
                        if let Some(text) = item["text"].as_str() {
                            text_parts.push(text.to_string());
                        } else if let Some(text) = item["content"].as_str() {
                            text_parts.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
            // Legacy / fallback: top-level assistant message
            Some("message") | Some("agent_message") => {
                if let Some(text) = event["content"].as_str() {
                    text_parts.push(text.to_string());
                } else if let Some(text) = event["text"].as_str() {
                    text_parts.push(text.to_string());
                }
            }
            // Streaming deltas — qgenie may emit these instead of item.completed
            Some("message_delta") | Some("content_block_delta") | Some("agent_message_content_delta") => {
                if let Some(text) = event["delta"]["text"].as_str() {
                    text_parts.push(text.to_string());
                } else if let Some(text) = event["delta"]["content"].as_str() {
                    text_parts.push(text.to_string());
                } else if let Some(text) = event["text"].as_str() {
                    text_parts.push(text.to_string());
                }
            }
            // Usage / turn summary — qgenie uses "turn.completed"
            Some("turn.completed") | Some("turn_complete") | Some("usage") => {
                let u = event.get("usage").unwrap_or(&event);
                let input = u["input_tokens"].as_u64().unwrap_or(0) as usize;
                let output_tokens = u["output_tokens"].as_u64().unwrap_or(0) as usize;
                let cached = u["cached_input_tokens"].as_u64().unwrap_or(0) as usize;
                if input > 0 || output_tokens > 0 {
                    usage = Some(AiUsage {
                        prompt_tokens: input,
                        completion_tokens: output_tokens,
                        total_tokens: input + output_tokens,
                        cached_tokens: if cached > 0 { Some(cached) } else { None },
                    });
                }
                // Some qgenie versions embed the last agent message in turn.completed
                if text_parts.is_empty() {
                    if let Some(text) = event["last_agent_message"].as_str() {
                        text_parts.push(text.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let response_text = text_parts.join("");
    if response_text.is_empty() {
        // Prefer the --output-last-message file content as a reliable fallback.
        if let Some(msg) = last_msg.filter(|s| !s.trim().is_empty()) {
            debug!("qgenie-cli: using --output-last-message fallback");
            return parse_inner_response(msg.trim(), usage);
        }
        // Last resort: treat the entire raw output as the response text.
        warn!("qgenie-cli: no recognised message events found, using raw output");
        return parse_inner_response(raw.trim(), usage);
    }

    parse_inner_response(&response_text, usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_item_completed_format() {
        // Real qgenie --json output format
        let raw = r#"{"type":"thread.started","thread_id":"abc123"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Hello world"}}
{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":0,"output_tokens":5}}
"#;
        let result = parse_qgenie_output(raw, None).unwrap();
        assert_eq!(result.content.as_deref(), Some("Hello world"));
        assert!(result.usage.is_some());
        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_parse_legacy_agent_message_format() {
        // Legacy top-level agent_message format
        let raw = r#"{"type":"agent_message","text":"Legacy response"}
{"type":"turn_complete","usage":{"input_tokens":8,"cached_input_tokens":0,"output_tokens":3}}
"#;
        let result = parse_qgenie_output(raw, None).unwrap();
        assert_eq!(result.content.as_deref(), Some("Legacy response"));
    }

    #[test]
    fn test_parse_multi_item_concatenation() {
        // Multiple item.completed events should be concatenated
        let raw = r#"{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}
{"type":"item.completed","item":{"type":"agent_message","text":" world"}}
{"type":"turn.completed","usage":{"input_tokens":5,"cached_input_tokens":0,"output_tokens":2}}
"#;
        let result = parse_qgenie_output(raw, None).unwrap();
        assert_eq!(result.content.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_parse_ignores_non_message_items() {
        // tool_call items should be ignored
        let raw = r#"{"type":"item.completed","item":{"type":"tool_call","name":"bash","input":{}}}
{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}
{"type":"turn.completed","usage":{"input_tokens":3,"cached_input_tokens":0,"output_tokens":1}}
"#;
        let result = parse_qgenie_output(raw, None).unwrap();
        assert_eq!(result.content.as_deref(), Some("Done"));
    }

    #[test]
    fn test_parse_output_last_message_fallback() {
        // When no agent_message items are present, fall back to --output-last-message content
        let raw = r#"{"type":"thread.started","thread_id":"abc123"}
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"mcp_tool_call","server":"s","tool":"t","result":{}}}
{"type":"turn.completed","usage":{"input_tokens":5,"cached_input_tokens":0,"output_tokens":2}}
"#;
        let last_msg = r#"{"concerns": []}"#;
        let result = parse_qgenie_output(raw, Some(last_msg)).unwrap();
        let content = result.content.unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(v.get("concerns").is_some());
    }

    #[test]
    fn test_parse_agent_message_content_delta() {
        // Streaming delta events should be collected
        let raw = r#"{"type":"turn.started"}
{"type":"agent_message_content_delta","delta":{"text":"Hello"}}
{"type":"agent_message_content_delta","delta":{"text":" world"}}
{"type":"turn.completed","usage":{"input_tokens":3,"cached_input_tokens":0,"output_tokens":2}}
"#;
        let result = parse_qgenie_output(raw, None).unwrap();
        assert_eq!(result.content.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_parse_turn_completed_last_agent_message() {
        // turn.completed may embed last_agent_message
        let raw = r#"{"type":"turn.started"}
{"type":"turn.completed","last_agent_message":"Embedded response","usage":{"input_tokens":2,"cached_input_tokens":0,"output_tokens":1}}
"#;
        let result = parse_qgenie_output(raw, None).unwrap();
        assert_eq!(result.content.as_deref(), Some("Embedded response"));
    }
}

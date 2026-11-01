use super::*;
fn chat(body: &str) -> Result<Request, ApiError> {
    parse_request(body.as_bytes(), true, "m")
}
fn completion(body: &str) -> Result<Request, ApiError> {
    parse_request(body.as_bytes(), false, "m")
}
#[test]
fn parses_chat_request() {
    let r = chat(
        r#"{"model":"m","messages":[{"role":"developer","content":"S"},
        {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
        {"role":"assistant","content":null,"reasoning_content":"r"},{"role":"user","content":"q"}],
        "max_tokens":5,"temperature":0.7,"top_p":0.9,"top_k":20,"seed":42,"stop":"END",
        "stream":true,"stream_options":{"include_usage":true},"n":1,"user":"u","logprobs":false,
        "presence_penalty":0,"frequency_penalty":null}"#,
    )
    .unwrap();
    let Prompt::Chat {
        messages,
        developer,
        ..
    } = &r.prompt
    else {
        panic!("expected chat prompt")
    };
    assert!(developer);
    assert_eq!(messages[0], Message::new(Role::System, "S"));
    assert_eq!(messages[1], Message::user("ab"));
    assert_eq!(messages[2].reasoning.as_deref(), Some("r"));
    assert_eq!(r.max_tokens, Some(5));
    assert_eq!(
        r.sampling,
        Sampling {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 20,
            seed: 42
        }
    );
    assert_eq!(r.stop, ["END"]);
    assert!(r.stream && r.include_usage);
    // Defaults: greedy, fill the context, no stop strings.
    let r =
        chat(r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":3}"#).unwrap();
    assert_eq!(r.max_tokens, Some(3));
    assert!(r.sampling.is_greedy() && !r.stream && r.stop.is_empty());
    let r = chat(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert_eq!(r.max_tokens, None);
}
#[test]
fn parses_completion_request() {
    let r = completion(r#"{"prompt":"Once","stop":["a","b"],"echo":false}"#).unwrap();
    assert_eq!(r.prompt, Prompt::Text("Once".into()));
    assert_eq!(r.max_tokens, Some(COMPLETIONS_DEFAULT_MAX_TOKENS));
    assert_eq!(r.stop, ["a", "b"]);
    for body in [
        r#"{"prompt":["a"]}"#,
        r#"{"prompt":""}"#,
        r#"{"messages":[]}"#,
        r#"{"prompt":"a","echo":true}"#,
        r#"{"prompt":"a","max_completion_tokens":2}"#,
    ] {
        assert_eq!(completion(body).unwrap_err().status, 400, "{body}");
    }
}
#[test]
fn rejects_unsupported_or_invalid_parameters() {
    let ok = r#""messages":[{"role":"user","content":"hi"}]"#;
    let with = |extra: &str| format!("{{{ok},{extra}}}");
    let cases = [
        ("not json".to_owned(), "invalid JSON"),
        ("[]".into(), "JSON object"),
        (r#"{"messages":[]}"#.into(), "must not be empty"),
        (r#"{"messages":"hi"}"#.into(), "must be an array"),
        (r#"{"max_tokens":2}"#.into(), "messages is required"),
        (with(r#""tools":[{"type":"x"}]"#), "tools[0]"),
        (with(r#""tool_choice":"required""#), "tool_choice"),
        (with(r#""response_format":{}"#), "response_format.type"),
        (with(r#""n":2"#), "n=1"),
        (with(r#""presence_penalty":0.5"#), "presence_penalty"),
        (with(r#""logprobs":true"#), "logprobs"),
        (with(r#""temperature":3"#), "temperature"),
        (
            with(r#""temperature":"hot""#),
            "temperature must be a number",
        ),
        (with(r#""top_p":0"#), "top_p"),
        (with(r#""top_k":-1"#), "top_k"),
        (with(r#""seed":1.5"#), "seed"),
        (with(r#""max_tokens":0"#), "at least 1"),
        (
            with(r#""max_tokens":2,"max_completion_tokens":2"#),
            "only one",
        ),
        (with(r#""stop":["a","b","c","d","e"]"#), "up to 4"),
        (with(r#""stop":[""]"#), "non-empty"),
        (with(r#""stop":1"#), "stop must be"),
        (
            with(r#""stream_options":{"include_usage":true}"#),
            "requires stream",
        ),
        (
            with(r#""stream":true,"stream_options":{"x":1}"#),
            "stream option",
        ),
        (with(r#""model":1"#), "model must be a string"),
        (
            r#"{"messages":[{"role":"function","content":"x"}]}"#.into(),
            "role 'function'",
        ),
        (
            r#"{"messages":[{"role":"user","content":"x","tool_calls":[]}]}"#.into(),
            "only valid on an assistant",
        ),
        (
            r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"[1]"}}]}]}"#.into(),
            "tool_calls must be",
        ),
        (
            r#"{"messages":[{"content":"x"}]}"#.into(),
            "role is required",
        ),
        (
            r#"{"messages":[{"role":"user"}]}"#.into(),
            "content is required",
        ),
        (
            r#"{"messages":[{"role":"user","content":"x","name":"n"}]}"#.into(),
            "name is not supported",
        ),
        (
            r#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#.into(),
            "image_url needs a url",
        ),
        (
            r#"{"messages":[{"role":"user","content":"x","reasoning_content":"r"}]}"#.into(),
            "assistant",
        ),
    ];
    for (body, expected) in cases {
        let e = chat(&body).unwrap_err();
        assert_eq!(e.status, 400, "{body}");
        assert!(e.message.contains(expected), "{body}: {}", e.message);
        assert_eq!(e.body()["error"]["type"], "invalid_request_error");
    }
    let e = chat(&with(r#""model":"other""#)).unwrap_err();
    assert_eq!(e.status, 404);
    assert_eq!(e.body()["error"]["type"], "not_found_error");
}
fn collect(stream: &mut TextStream, pieces: &[&[u8]]) -> Vec<(bool, String)> {
    let mut out: Vec<Delta> = pieces.iter().flat_map(|p| stream.push(p)).collect();
    out.extend(stream.finish());
    out.into_iter().map(|d| (d.reasoning, d.text)).collect()
}
#[test]
fn text_stream_splits_reasoning_and_stops() {
    let mut s = TextStream::new(true, vec!["STOP".into()], false);
    let out = collect(
        &mut s,
        &[b"thin", b"k</th", b"ink>Hel", b"lo ST", b"OP more"],
    );
    let expected = [(true, "thin"), (true, "k"), (false, "Hel"), (false, "lo ")];
    assert_eq!(out, expected.map(|(r, t)| (r, t.to_owned())));
    assert!(s.stopped);
    // A partial stop match that does not complete is released.
    let mut s = TextStream::new(false, vec!["STOP".into()], false);
    let out = collect(&mut s, &[b"a ST", b"ar"]);
    assert_eq!(out, [(false, "a ".into()), (false, "STar".into())]);
    assert!(!s.stopped);
    // Multi-byte characters split across tokens are joined before matching.
    let mut s = TextStream::new(false, vec!["。".into()], false);
    let bytes = "好。".as_bytes();
    let out = collect(&mut s, &[&bytes[..2], &bytes[2..4], &bytes[4..]]);
    assert_eq!(out, [(false, "好".into())]);
    // Incomplete UTF-8 at the end is flushed as U+FFFD.
    let mut s = TextStream::new(false, vec![], false);
    let out = collect(&mut s, &[b"ok", &[0xe4, 0xbd]]);
    assert_eq!(out, [(false, "ok".into()), (false, "\u{fffd}".into())]);
    // The earliest stop string wins.
    let mut s = TextStream::new(false, vec!["b".into(), "a".into()], false);
    assert_eq!(collect(&mut s, &[b"xab"]), [(false, "x".into())]);
    assert_eq!(s.matched.as_deref(), Some("a"));
    // With tools, text from <tool_call> on is captured; stops no longer apply.
    let mut s = TextStream::new(false, vec!["END".into()], true);
    let out = collect(
        &mut s,
        &[b"Let me check.<tool", b"_call>\n<function=f>END", b"\n"],
    );
    assert_eq!(out, [(false, "Let me check.".into())]);
    assert_eq!(s.tool_text, "<tool_call>\n<function=f>END\n");
    assert!(!s.stopped);
    // Without tools the marker is ordinary text.
    let mut s = TextStream::new(false, vec![], false);
    assert_eq!(
        collect(&mut s, &[b"<tool_call>"]),
        [(false, "<tool_call>".into())]
    );
}
#[test]
fn parses_tools_and_tool_messages() {
    let r = chat(
        r#"{"tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
        "tool_choice":"auto","parallel_tool_calls":false,"messages":[{"role":"user","content":"q"},
        {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",
          "function":{"name":"f","arguments":"{\"x\":1}"}}]},
        {"role":"tool","tool_call_id":"c1","content":"r"}]}"#,
    )
    .unwrap();
    assert_eq!(r.tools.len(), 1);
    assert!(!r.parallel_tool_calls);
    let Prompt::Chat { messages, .. } = &r.prompt else {
        panic!("expected chat prompt")
    };
    assert_eq!(messages[1].tool_calls[0].arguments["x"], 1);
    assert_eq!(messages[2].role, Role::Tool);
    let none = chat(r#"{"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"none","messages":[{"role":"user","content":"q"}]}"#).unwrap();
    assert!(none.tools.is_empty());
}
#[test]
fn parses_thinking_levels() {
    let level = |extra: &str| {
        chat(&format!(
            r#"{{"messages":[{{"role":"user","content":"q"}}]{extra}}}"#
        ))
        .map(|r| r.thinking)
    };
    assert_eq!(level(""), Ok(None));
    assert_eq!(
        level(r#","reasoning_effort":"none""#),
        Ok(Some(Thinking::Off))
    );
    assert_eq!(
        level(r#","reasoning_effort":"low""#),
        Ok(Some(Thinking::Low))
    );
    assert_eq!(
        level(r#","reasoning_effort":"medium""#),
        Ok(Some(Thinking::Medium))
    );
    assert_eq!(
        level(r#","reasoning_effort":"xhigh""#),
        Ok(Some(Thinking::High))
    );
    let kwargs = |v: &str| {
        level(&format!(
            r#","chat_template_kwargs":{{"enable_thinking":{v}}}"#
        ))
    };
    assert_eq!(kwargs("false"), Ok(Some(Thinking::Off)));
    assert_eq!(kwargs("true"), Ok(Some(Thinking::High)));
    assert_eq!(
        level(r#","chat_template_kwargs":{"enable_thinking":true},"reasoning_effort":"low""#),
        Ok(Some(Thinking::Low))
    );
    for extra in [
        r#","reasoning_effort":"max""#,
        r#","reasoning_effort":1"#,
        r#","chat_template_kwargs":{"enable_thinking":"yes"}"#,
        r#","chat_template_kwargs":{"thinking_budget":5}"#,
        r#","chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"high""#,
        r#","chat_template_kwargs":{"enable_thinking":true},"reasoning_effort":"none""#,
    ] {
        assert_eq!(level(extra).unwrap_err().status, 400, "{extra}");
    }
    assert!(completion(r#"{"prompt":"a","reasoning_effort":"low"}"#).is_err());
}
#[test]
fn text_stream_routes_a_think_block_the_reply_opens() {
    // Qwen3 with enable_thinking: the reply opens the block itself.
    let mut s = TextStream::new(false, vec![], true).detect_think();
    let out = collect(
        &mut s,
        &[b"<thi", b"nk>\nplan\n</think>\n\nOK", b"<tool_call>\nx"],
    );
    assert_eq!(out, [(true, "plan\n".into()), (false, "OK".into())]);
    assert_eq!(s.tool_text, "<tool_call>\nx");
    // A reply without a think block is the answer.
    let mut s = TextStream::new(false, vec![], false).detect_think();
    assert_eq!(
        collect(&mut s, &[b"<th", b"e end"]),
        [(false, "<the end".into())]
    );
    // An open think block in the prompt (Qwen3.8): whitespace around it is dropped,
    // and a tool call after the reasoning is still captured.
    let mut s = TextStream::new(true, vec![], true);
    let out = collect(
        &mut s,
        &[b"\nr", b"\n</think>\n", b"\n<tool_call>\n<function=f>"],
    );
    assert_eq!(out, [(true, "r".into()), (true, "\n".into())]);
    assert_eq!(s.tool_text, "<tool_call>\n<function=f>");
}

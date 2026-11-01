use super::*;
fn sockets() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    (client, listener.accept().unwrap().0)
}
fn completion_request(client: &mut TcpStream, streaming: bool) {
    let body = json!({"prompt": "hello", "stream": streaming}).to_string();
    write!(
        client,
        "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
}
fn response_from_worker(streaming: bool, started: bool) -> String {
    let (mut client, server) = sockets();
    let (jobs, queue) = mpsc::sync_channel(MAX_PENDING);
    let shared = Arc::new(Shared::new(jobs, Served::Fixed("test".into())));
    let connection = std::thread::spawn(move || handle(server, shared));
    completion_request(&mut client, streaming);
    let job = queue.recv_timeout(Duration::from_secs(5)).unwrap();
    if started {
        job.events
            .send(Event::Started { prompt_tokens: 0 })
            .unwrap();
    }
    job.events
        .send(Event::Error(ApiError::new(500, "test generation failure")))
        .unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    connection.join().unwrap();
    response
}
#[test]
fn streaming_failure_is_an_sse_error_after_headers() {
    let response = response_from_worker(true, true);
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    assert_eq!(response.matches("HTTP/1.1").count(), 1, "{response}");
    let body = response.split_once("\r\n\r\n").unwrap().1;
    let data = body.strip_prefix("data: ").unwrap().trim();
    let error: Value = serde_json::from_str(data).unwrap();
    assert_eq!(error["error"]["message"], "test generation failure");
    assert!(!body.contains("[DONE]"));
}
#[test]
fn failures_before_stream_headers_and_nonstreaming_use_http_status() {
    for (streaming, started) in [(true, false), (false, true)] {
        let response = response_from_worker(streaming, started);
        assert!(
            response.starts_with("HTTP/1.1 500 Internal Server Error\r\n"),
            "{response}"
        );
        let body = response.split_once("\r\n\r\n").unwrap().1;
        let error: Value = serde_json::from_str(body).unwrap();
        assert_eq!(error["error"]["message"], "test generation failure");
    }
}
#[test]
fn full_job_queue_rejects_immediately_and_recovers() {
    let (jobs, queue) = mpsc::sync_channel(MAX_PENDING);
    let shared = Arc::new(Shared::new(jobs, Served::Fixed("test".into())));
    for _ in 0..MAX_PENDING {
        let (events, _) = mpsc::sync_channel(MAX_EVENTS);
        shared
            .jobs
            .try_send(Job {
                model: "test".into(),
                chat: false,
                request: parse_request(br#"{"prompt":"queued"}"#, false, "test").unwrap(),
                cancel: Cancellation::default(),
                events,
            })
            .ok()
            .unwrap();
    }
    let (mut client, server) = sockets();
    completion_request(&mut client, true);
    handle(server, shared.clone());
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(
        response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
        "{response}"
    );
    for _ in 0..MAX_PENDING {
        queue.try_recv().unwrap();
    }
    assert!(matches!(queue.try_recv(), Err(mpsc::TryRecvError::Empty)));
    let (mut client, server) = sockets();
    let connection = std::thread::spawn(move || handle(server, shared));
    completion_request(&mut client, false);
    let job = queue.recv_timeout(Duration::from_secs(5)).unwrap();
    job.events
        .send(Event::Started { prompt_tokens: 0 })
        .unwrap();
    job.events
        .send(Event::Done {
            finish: "stop",
            prompt_tokens: 1,
            completion_tokens: 0,
            tool_calls: vec![],
            stop_sequence: None,
        })
        .unwrap();
    response.clear();
    client.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    connection.join().unwrap();
}
#[test]
fn connection_limit_rejects_and_releases_finished_handlers() {
    let (jobs, _queue) = mpsc::sync_channel(MAX_PENDING);
    let shared = Arc::new(Shared::new(jobs, Served::Fixed("test".into())));
    let mut connections = Vec::new();
    let mut clients = Vec::new();
    for _ in 0..MAX_CONNECTIONS {
        let (client, server) = sockets();
        clients.push(client);
        accept_connection(server, &shared, &mut connections);
    }
    let (mut client, server) = sockets();
    accept_connection(server, &shared, &mut connections);
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(
        response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
        "{response}"
    );
    assert_eq!(connections.len(), MAX_CONNECTIONS);
    // End the requests and read their responses, so the handlers finish.
    for mut client in clients {
        client
            .write_all(b"GET /v1/models HTTP/1.1\r\n\r\n")
            .unwrap();
        client.read_to_end(&mut Vec::new()).unwrap();
    }
    // Join one handler; the remaining finished handles exercise pruning.
    connections.pop().unwrap().join().unwrap();
    let (mut client, server) = sockets();
    client
        .write_all(b"GET /v1/models HTTP/1.1\r\n\r\n")
        .unwrap();
    accept_connection(server, &shared, &mut connections);
    response.clear();
    client.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    for connection in connections {
        connection.join().unwrap();
    }
}
#[test]
fn disconnect_releases_a_worker_sending_to_a_slow_stream() {
    let (mut client, server) = sockets();
    let (jobs, queue) = mpsc::sync_channel(MAX_PENDING);
    let shared = Arc::new(Shared::new(jobs, Served::Fixed("test".into())));
    let connection = std::thread::spawn(move || handle(server, shared));
    completion_request(&mut client, true);
    let job = queue.recv_timeout(Duration::from_secs(5)).unwrap();
    let cancel = job.cancel.clone();
    let (done, finished) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        job.events
            .send(Event::Started { prompt_tokens: 0 })
            .unwrap();
        while job
            .events
            .send(Event::Delta(Delta {
                reasoning: false,
                text: "x".repeat(64 * 1024),
            }))
            .is_ok()
        {}
        done.send(()).unwrap();
    });
    // Read only the response header, then stop consuming generated text.
    let mut header = Vec::new();
    while !header.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        client.read_exact(&mut byte).unwrap();
        header.push(byte[0]);
    }
    client.shutdown(std::net::Shutdown::Both).unwrap();
    finished.recv_timeout(Duration::from_secs(5)).unwrap();
    worker.join().unwrap();
    connection.join().unwrap();
    assert!(cancel.is_cancelled());
}
#[test]
fn root_serves_the_offline_chat_page() {
    let (jobs, _queue) = mpsc::sync_channel(MAX_PENDING);
    let shared = Arc::new(Shared::new(jobs, Served::Fixed("test".into())));
    let (mut client, server) = sockets();
    client.write_all(b"GET / HTTP/1.1\r\n\r\n").unwrap();
    handle(server, shared);
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200 OK\r\nContent-Type: text/html"));
    assert!(response.ends_with(WEB_PAGE));
    // Only same-origin requests: the page must work offline.
    assert!(!WEB_PAGE.contains("http://") && !WEB_PAGE.contains("https://"));
}
#[test]
fn file_id_strips_shard_suffix() {
    assert_eq!(
        catalog::file_id("/m/Qwen-IQ2_XS-00001-of-00002.gguf".as_ref()),
        "Qwen-IQ2_XS"
    );
    assert_eq!(catalog::file_id("/m/single.gguf".as_ref()), "single");
}
fn exchange(raw: &[u8]) -> std::result::Result<HttpRequest, ApiError> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client.write_all(raw).unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    read_request(&mut server)
}
#[test]
fn reads_http_requests_and_rejects_bad_framing() {
    let r =
        exchange(b"POST /v1/completions?x=1 HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody").unwrap();
    assert_eq!(
        (r.method.as_str(), r.path.as_str()),
        ("POST", "/v1/completions")
    );
    assert_eq!(r.body, b"body");
    assert_eq!(
        exchange(b"GET /v1/models HTTP/1.1\r\n\r\n").unwrap().body,
        b""
    );
    let status = |raw: &[u8]| exchange(raw).err().map(|e| e.status);
    assert_eq!(status(b"POST /v1/completions HTTP/1.1\r\n\r\n"), Some(411));
    assert_eq!(
        status(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"),
        Some(501)
    );
    assert_eq!(
        status(b"POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n"),
        Some(413)
    );
    assert_eq!(
        status(b"POST / HTTP/1.1\r\nContent-Length: 9\r\n\r\nshort"),
        Some(400)
    );
    assert_eq!(status(b"GARBAGE\r\n\r\n"), Some(400));
    assert_eq!(status(b"GET / HTTP/2\r\n\r\n"), Some(400));
}
fn get(shared: Arc<Shared>, request: &str) -> String {
    let (mut client, server) = sockets();
    client.write_all(request.as_bytes()).unwrap();
    handle(server, shared);
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    response
}
#[test]
fn api_key_guards_api_routes_but_not_the_page() {
    let (jobs, _queue) = mpsc::sync_channel(MAX_PENDING);
    let mut shared = Shared::new(jobs, Served::Fixed("test".into()));
    shared.api_key = Some("secret".into());
    let shared = Arc::new(shared);
    let page = get(shared.clone(), "GET / HTTP/1.1\r\n\r\n");
    assert!(page.starts_with("HTTP/1.1 200 OK\r\n"), "{page}");
    for (request, kind) in [
        ("GET /v1/models HTTP/1.1\r\n\r\n", "invalid_request_error"),
        (
            "GET /metrics HTTP/1.1\r\nAuthorization: Bearer wrong\r\n\r\n",
            "invalid_request_error",
        ),
        (
            "POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            "authentication_error",
        ),
    ] {
        let response = get(shared.clone(), request);
        assert!(
            response.starts_with("HTTP/1.1 401 Unauthorized\r\n"),
            "{response}"
        );
        let body: Value = serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["error"]["type"], kind, "{request}");
    }
    for header in ["Authorization: bearer secret", "x-api-key: secret"] {
        let response = get(
            shared.clone(),
            &format!("GET /v1/models HTTP/1.1\r\n{header}\r\n\r\n"),
        );
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    }
    let metrics = get(
        shared.clone(),
        "GET /metrics HTTP/1.1\r\nx-api-key: secret\r\n\r\n",
    );
    assert!(
        metrics.contains("\r\n\r\n# HELP ingot_requests_total"),
        "{metrics}"
    );
    let status = get(
        shared,
        "GET /v1/status HTTP/1.1\r\nx-api-key: secret\r\n\r\n",
    );
    let body: Value = serde_json::from_str(status.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["queued"], 0);
}

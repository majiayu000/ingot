use super::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

fn sha(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ingot-models-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn manifest(entries: &[(&str, &str, &[u8])]) -> Manifest {
    let mut text = String::new();
    for (model, file, data) in entries {
        text += &format!(
            "[{model}]\nkind = main\nsummary = test\nresident_bytes = 1\nfile = org/repo {} dir/{file} {} {}\n",
            "a".repeat(40),
            data.len(),
            sha(data)
        );
    }
    Manifest::parse(&text).unwrap()
}

/// Serves `data` to `requests` connections, honouring Range only when
/// `ranges`; records each request line and requested offset.
fn serve(data: Vec<u8>, ranges: bool, requests: usize) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(requests) {
            let mut stream = stream.unwrap();
            let mut head = Vec::new();
            let mut byte = [0];
            while !head.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                head.push(byte[0]);
            }
            let head = String::from_utf8(head).unwrap();
            let range = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("range: bytes=")
                        .map(String::from)
                })
                .and_then(|r| r.trim_end_matches('-').parse::<usize>().ok());
            let line = head.lines().next().unwrap().to_string();
            log.lock()
                .unwrap()
                .push(line + &range.map(|r| format!(" range={r}")).unwrap_or_default());
            let (status, start) = match range.filter(|_| ranges) {
                Some(start) => (
                    format!(
                        "206 Partial Content\r\nContent-Range: bytes {start}-{}/{}",
                        data.len() - 1,
                        data.len()
                    ),
                    start,
                ),
                None => ("200 OK".to_string(), 0),
            };
            let mut response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                data.len() - start
            )
            .into_bytes();
            response.extend_from_slice(&data[start..]);
            stream.write_all(&response).unwrap();
        }
    });
    (endpoint, seen)
}

/// Port 9 (discard) is closed locally, so any request fails to connect.
const NO_SERVER: &str = "http://127.0.0.1:9";

#[test]
fn builtin_manifest_parses_and_shares_qwen_shard_two() {
    let m = Manifest::builtin().unwrap();
    let iq2 = m.get("qwen3.8-iq2_xs").unwrap();
    let iq3 = m.get("qwen3.8-iq3_xxs").unwrap();
    assert_eq!(iq2.files[1].sha256, iq3.files[1].sha256);
    assert_eq!(
        iq2.files[1].sha256,
        "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
    );
    assert_eq!(iq2.draft.as_deref(), Some("qwen3.8-mtp-q8_0"));
    assert_eq!(m.get("glm5.3-mtp-q8_0").unwrap().kind, Kind::Draft);
    assert!(m.get("missing").is_err());
}

#[test]
fn manifest_rejects_malformed_entries() {
    let file = format!(
        "file = org/repo {} a.gguf 1 {}",
        "a".repeat(40),
        "b".repeat(64)
    );
    let ok = format!("[m]\nkind = main\nsummary = s\nresident_bytes = 1\n{file}\n");
    assert!(Manifest::parse(&ok).is_ok());
    for bad in [
        String::new(),
        format!("kind = main\n{ok}"),
        format!("{ok}[m]\n"),
        ok.replace("kind = main", "kind = other"),
        ok.replace("summary = s\n", ""),
        ok.replace("resident_bytes = 1", "resident_bytes = x"),
        ok.replace(&"b".repeat(64), &"B".repeat(64)),
        ok.replace(" 1 ", " 0 "),
        ok.replace("org/repo ", "org/repo extra "),
        format!("{ok}{file}\n"),
        format!("{ok}draft = m\n"),
        format!("{ok}bogus = 1\n"),
        format!("{ok}kind = draft\n"),
    ] {
        assert!(Manifest::parse(&bad).is_err(), "{bad}");
    }
}

#[test]
fn recommendation_caps_at_working_set() {
    assert_eq!(
        recommend(40 * GIB, 72 * GIB),
        Budget {
            gib: 44,
            resident: true
        }
    );
    assert_eq!(recommend(40 * GIB + 1, 72 * GIB).gib, 45);
    assert!(!recommend(65 * GIB, 72 * GIB).resident);
    assert_eq!(
        recommend(110 * GIB, 72 * GIB + 5),
        Budget {
            gib: 68,
            resident: false
        }
    );
}

#[test]
fn download_resumes_from_partial_file() {
    let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
    let m = manifest(&[("m", "f.gguf", &data)]);
    let root = temp("resume");
    fs::create_dir_all(root.join("m")).unwrap();
    fs::write(root.join("m/f.gguf.part"), &data[..123_456]).unwrap();
    let (endpoint, seen) = serve(data.clone(), true, 1);
    let path = pull(&m, "m", &root, &endpoint).unwrap();
    assert_eq!(path, root.join("m/f.gguf"));
    assert_eq!(fs::read(&path).unwrap(), data);
    assert!(!root.join("m/f.gguf.part").exists());
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let expected = format!("GET /org/repo/resolve/{}/dir/f.gguf ", "a".repeat(40));
    assert!(seen[0].starts_with(&expected), "{}", seen[0]);
    assert!(seen[0].ends_with("range=123456"), "{}", seen[0]);
}

#[test]
fn download_restarts_when_server_ignores_range() {
    let data = b"hello resumable world".to_vec();
    let m = manifest(&[("m", "f.gguf", &data)]);
    let root = temp("norange");
    fs::create_dir_all(root.join("m")).unwrap();
    fs::write(root.join("m/f.gguf.part"), b"hello").unwrap();
    let (endpoint, _) = serve(data.clone(), false, 1);
    let path = pull(&m, "m", &root, &endpoint).unwrap();
    assert_eq!(fs::read(path).unwrap(), data);
}

#[test]
fn hash_mismatch_fails_and_removes_partial() {
    let m = manifest(&[("m", "f.gguf", b"expected")]);
    let root = temp("mismatch");
    let (endpoint, _) = serve(b"tampered".to_vec(), true, 1);
    let error = pull(&m, "m", &root, &endpoint).unwrap_err();
    assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");
    assert!(!root.join("m/f.gguf").exists());
    assert!(!root.join("m/f.gguf.part").exists());
}

#[test]
fn truncated_download_keeps_partial_for_resume() {
    let data = b"0123456789".to_vec();
    let m = manifest(&[("m", "f.gguf", &data)]);
    let root = temp("truncated");
    // The server announces and sends only 4 bytes of a 10-byte file.
    let (endpoint, _) = serve(data[..4].to_vec(), true, 1);
    let error = pull(&m, "m", &root, &endpoint).unwrap_err();
    assert!(
        error.to_string().contains("rerun pull to resume"),
        "{error}"
    );
    assert_eq!(fs::read(root.join("m/f.gguf.part")).unwrap(), &data[..4]);
    let (endpoint, seen) = serve(data.clone(), true, 1);
    assert_eq!(
        fs::read(pull(&m, "m", &root, &endpoint).unwrap()).unwrap(),
        data
    );
    assert!(seen.lock().unwrap()[0].ends_with("range=4"));
}

#[test]
fn existing_files_are_verified_without_network() {
    let data = b"present".to_vec();
    let m = manifest(&[("m", "f.gguf", &data)]);
    let root = temp("existing");
    fs::create_dir_all(root.join("custom")).unwrap();
    fs::write(root.join("custom/f.gguf"), &data).unwrap();
    let path = pull(&m, "m", &root, NO_SERVER).unwrap();
    assert_eq!(path, root.join("custom/f.gguf"));

    fs::write(root.join("custom/f.gguf"), b"PRESENT").unwrap();
    let error = pull(&m, "m", &root, NO_SERVER).unwrap_err();
    assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");
}

#[test]
fn identical_shards_are_hard_linked() {
    use std::os::unix::fs::MetadataExt;
    let data = b"shared shard".to_vec();
    let m = manifest(&[("a", "a.gguf", &data), ("b", "b.gguf", &data)]);
    let root = temp("link");
    fs::create_dir_all(root.join("a")).unwrap();
    fs::write(root.join("a/a.gguf"), &data).unwrap();
    let path = pull(&m, "b", &root, NO_SERVER).unwrap();
    assert_eq!(path, root.join("b/b.gguf"));
    assert_eq!(
        fs::metadata(&path).unwrap().ino(),
        fs::metadata(root.join("a/a.gguf")).unwrap().ino()
    );
}

#[test]
fn network_errors_are_reported() {
    let m = manifest(&[("m", "f.gguf", b"x")]);
    let root = temp("network");
    let error = pull(&m, "m", &root, NO_SERVER).unwrap_err();
    assert!(
        error.to_string().contains("rerun pull to resume"),
        "{error}"
    );
    assert!(!root.join("m/f.gguf").exists());
}

#[test]
fn list_reports_present_missing_and_wrong_size() {
    let m = manifest(&[("a", "a.gguf", b"aa"), ("b", "b.gguf", b"bb")]);
    let root = temp("list");
    fs::create_dir_all(root.join("x/.hidden")).unwrap();
    fs::write(root.join("x/a.gguf"), b"aa").unwrap();
    fs::write(root.join("x/b.gguf"), b"b").unwrap();
    fs::write(root.join("x/.hidden/b.gguf"), b"bb").unwrap();
    let text = list(&m, &root, "test", 72 * GIB).unwrap();
    let line = |name: &str| text.lines().find(|l| l.starts_with(name)).unwrap();
    assert!(line("a ").contains("present"), "{text}");
    assert!(line("b ").contains("missing"), "{text}");
    assert!(text.contains("expected 2"), "{text}");
}

#[test]
fn discovery_requires_colocated_shards_and_prefers_complete_copy() {
    let mut m = manifest(&[
        ("m", "first.gguf", b"first"),
        ("other", "second.gguf", b"second"),
    ]);
    let second = m.models.pop().unwrap().files.remove(0);
    m.models[0].files.push(second);
    let root = temp("colocated");
    for dir in ["a", "b", "z"] {
        fs::create_dir_all(root.join(dir)).unwrap();
    }
    fs::write(root.join("a/first.gguf"), b"first").unwrap();
    fs::write(root.join("b/second.gguf"), b"second").unwrap();
    assert!(resolve(&m, "m", &root).is_err());
    let text = list(&m, &root, "test", 72 * GIB).unwrap();
    let row = text.lines().find(|l| l.starts_with("m ")).unwrap();
    assert!(row.contains("partial 1/2"), "{text}");

    fs::write(root.join("z/first.gguf"), b"first").unwrap();
    fs::write(root.join("z/second.gguf"), b"second").unwrap();
    assert_eq!(
        resolve(&m, "m", &root).unwrap().model,
        root.join("z/first.gguf")
    );
    let text = list(&m, &root, "test", 72 * GIB).unwrap();
    assert!(
        text.contains(&format!("m: {}", root.join("z/first.gguf").display())),
        "{text}"
    );
    assert_eq!(
        pull(&m, "m", &root, NO_SERVER).unwrap(),
        root.join("z/first.gguf")
    );
    assert!(!root.join("a/second.gguf").exists());
}

#[test]
fn pull_assembles_scattered_shards_without_redownloading() {
    let mut m = manifest(&[
        ("m", "first.gguf", b"first"),
        ("other", "second.gguf", b"second"),
    ]);
    let second = m.models.pop().unwrap().files.remove(0);
    m.models[0].files.push(second);
    let root = temp("assemble");
    for dir in ["a", "b"] {
        fs::create_dir_all(root.join(dir)).unwrap();
    }
    fs::write(root.join("a/first.gguf"), b"first").unwrap();
    fs::write(root.join("b/second.gguf"), b"second").unwrap();
    let path = pull(&m, "m", &root, NO_SERVER).unwrap();
    assert_eq!(path, root.join("a/first.gguf"));
    assert_eq!(fs::read(root.join("a/second.gguf")).unwrap(), b"second");
    assert_eq!(resolve(&m, "m", &root).unwrap().model, path);
}

#[test]
fn resolve_finds_names_paths_and_drafts() {
    let h = "a".repeat(40);
    let entry = |name: &str, kind: &str, extra: &str, file: &str, data: &[u8]| {
        format!(
            "[{name}]\nkind = {kind}\nsummary = t\nresident_bytes = 1\n{extra}file = org/repo {h} {file} {} {}\n",
            data.len(),
            sha(data)
        )
    };
    let m = Manifest::parse(
        &(entry("m", "main", "draft = d\n", "m1.gguf", b"m1")
            + &entry("d", "draft", "", "d.gguf", b"dd")),
    )
    .unwrap();
    let root = temp("resolve");
    let err = resolve(&m, "m", &root).unwrap_err().to_string();
    assert!(err.contains("ingot pull m"), "{err}");
    fs::create_dir_all(root.join("x")).unwrap();
    fs::write(root.join("x/m1.gguf"), b"m1").unwrap();
    let r = resolve(&m, "m", &root).unwrap();
    assert_eq!((r.model, r.draft), (root.join("x/m1.gguf"), None));
    fs::write(root.join("x/d.gguf"), b"dd").unwrap();
    let path = root.join("x/m1.gguf");
    let r = resolve(&m, path.to_str().unwrap(), &root).unwrap();
    assert_eq!(r.draft, Some(root.join("x/d.gguf")));
    assert!(resolve(&m, "d", &root).is_err());
    assert!(resolve(&m, "nope", &root).is_err());
}

#[test]
#[ignore = "requires INGOT_MODELS with local model files"]
fn manifest_resident_bytes_match_local_files() {
    let m = Manifest::builtin().unwrap();
    let found = locate(&default_dir().unwrap(), &m.file_names()).unwrap();
    let mut mismatched = Vec::new();
    for model in &m.models {
        let Some(path) = sized(&found, &model.files[0]) else {
            println!("{}: not present", model.name);
            continue;
        };
        // Drafts are reserved by file size; main models by resident tensors.
        let bytes = match model.kind {
            Kind::Main => resident_weight_bytes(&Gguf::open(path).unwrap()).unwrap(),
            Kind::Draft => model.download_bytes(),
        };
        println!("{} resident_bytes = {bytes}", model.name);
        if bytes != model.resident_bytes {
            mismatched.push(&model.name);
        }
    }
    assert!(mismatched.is_empty(), "{mismatched:?}");
}

#[test]
#[ignore = "network: downloads one small file from Hugging Face"]
fn real_hugging_face_small_file() {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.into());
    let text = std::env::var("INGOT_TEST_FILE")
        .expect("INGOT_TEST_FILE='REPO REVISION PATH SIZE SHA256' of a small (<100 MB) file");
    let m = Manifest::parse(&format!(
        "[small]\nkind = main\nsummary = s\nresident_bytes = 1\nfile = {text}\n"
    ))
    .unwrap();
    let root = temp("real");
    let path = pull(&m, "small", &root, &endpoint).unwrap();
    // Truncate and resume against the real server and its CDN redirect.
    let file = &m.models[0].files[0];
    let data = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    let part = path.with_file_name(format!("{}.part", file.name()));
    fs::write(part, &data[..data.len() / 2]).unwrap();
    assert_eq!(
        fs::read(pull(&m, "small", &root, &endpoint).unwrap()).unwrap(),
        data
    );
}

/// Answers `GET PATH` from `routes` (status, body) and 404 otherwise, for
/// `requests` connections; records each request path.
fn serve_routes(
    routes: Vec<(String, u16, Vec<u8>)>,
    requests: usize,
) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(requests) {
            let mut stream = stream.unwrap();
            let mut head = Vec::new();
            let mut byte = [0];
            while !head.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                head.push(byte[0]);
            }
            let head = String::from_utf8(head).unwrap();
            let path = head.split_whitespace().nth(1).unwrap().to_string();
            log.lock().unwrap().push(path.clone());
            let (status, body) = routes
                .iter()
                .find(|(p, ..)| *p == path)
                .map_or((404, b"{}".to_vec()), |(_, s, b)| (*s, b.clone()));
            let mut response = format!(
                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .into_bytes();
            response.extend_from_slice(&body);
            stream.write_all(&response).unwrap();
        }
    });
    (endpoint, seen)
}

/// Tree listing route for `dir` in org/repo at main plus one resolve route
/// per `(path, listed data, served data)`.
fn hf_routes(dir: &str, files: &[(&str, &[u8], &[u8])]) -> Vec<(String, u16, Vec<u8>)> {
    let entries: Vec<_> = files
        .iter()
        .map(|(path, listed, _)| {
            serde_json::json!({"type": "file", "oid": "0".repeat(40), "size": listed.len(), "path": path,
                "lfs": {"oid": sha(listed), "size": listed.len(), "pointerSize": 134}})
        })
        .collect();
    let tree = format!("/api/models/org/repo/tree/main{dir}");
    let mut routes = vec![(tree, 200, serde_json::to_vec(&entries).unwrap())];
    for (path, _, served) in files {
        routes.push((
            format!("/org/repo/resolve/main/{path}"),
            200,
            served.to_vec(),
        ));
    }
    routes
}

#[test]
fn hf_spec_parses_repo_path_revision_and_shards() {
    let spec = |s: &str| hf::Spec::parse(s).unwrap();
    assert!(hf::Spec::parse("qwen3.8-iq2_xs").is_none());
    let s = spec("hf:org/repo/sub/m-Q4.gguf@v1.0").unwrap();
    assert_eq!(
        (s.repo.as_str(), s.revision.as_str(), s.path.as_str()),
        ("org/repo", "v1.0", "sub/m-Q4.gguf")
    );
    assert_eq!(spec("hf:org/repo/m.gguf").unwrap().revision, "main");
    assert_eq!(
        spec("hf:org/repo/d/m-00002-of-00003.gguf")
            .unwrap()
            .shard_paths(),
        [
            "d/m-00001-of-00003.gguf",
            "d/m-00002-of-00003.gguf",
            "d/m-00003-of-00003.gguf"
        ]
    );
    for bad in [
        "hf:org/m.gguf",
        "hf:org/repo/m.bin",
        "hf:org/repo/../m.gguf",
        "hf:org//repo/m.gguf",
        "hf:org/repo/m.gguf@",
    ] {
        assert!(spec(bad).is_err(), "{bad}");
    }
}

#[test]
fn hf_pull_downloads_verifies_and_resolves() {
    let m = manifest(&[("m", "f.gguf", b"x")]);
    let data: Vec<u8> = (0..50_000u32).map(|i| (i * 3) as u8).collect();
    let root = temp("hf-single");
    let name = "hf:org/repo/f.gguf";
    let err = resolve(&m, name, &root).unwrap_err().to_string();
    assert!(err.contains("ingot pull hf:org/repo/f.gguf"), "{err}");
    let (endpoint, seen) = serve_routes(hf_routes("", &[("f.gguf", &data, &data)]), 2);
    let path = pull(&m, name, &root, &endpoint).unwrap();
    assert_eq!(path, root.join("hf/org/repo/f.gguf"));
    assert_eq!(fs::read(&path).unwrap(), data);
    assert_eq!(
        *seen.lock().unwrap(),
        [
            "/api/models/org/repo/tree/main",
            "/org/repo/resolve/main/f.gguf"
        ]
    );
    assert_eq!(resolve(&m, name, &root).unwrap().model, path);
    // A second pull only lists the tree and verifies the local copy.
    let (endpoint, seen) = serve_routes(hf_routes("", &[("f.gguf", &data, &data)]), 1);
    assert_eq!(pull(&m, name, &root, &endpoint).unwrap(), path);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn hf_pull_fetches_every_shard() {
    let m = manifest(&[("m", "f.gguf", b"x")]);
    let (one, two) = (b"first shard".as_slice(), b"second shard".as_slice());
    let files = [
        ("q/m-00001-of-00002.gguf", one, one),
        ("q/m-00002-of-00002.gguf", two, two),
    ];
    let root = temp("hf-shards");
    let name = "hf:org/repo/q/m-00002-of-00002.gguf";
    let (endpoint, _) = serve_routes(hf_routes("/q", &files), 3);
    let path = pull(&m, name, &root, &endpoint).unwrap();
    assert_eq!(path, root.join("hf/org/repo/q/m-00001-of-00002.gguf"));
    assert_eq!(
        fs::read(root.join("hf/org/repo/q/m-00002-of-00002.gguf")).unwrap(),
        two
    );
    assert_eq!(resolve(&m, name, &root).unwrap().model, path);
    fs::remove_file(root.join("hf/org/repo/q/m-00002-of-00002.gguf")).unwrap();
    assert!(resolve(&m, name, &root).is_err());
}

#[test]
fn hf_pull_reports_missing_files_mismatches_and_http_errors() {
    let m = manifest(&[("m", "f.gguf", b"x")]);
    let root = temp("hf-errors");
    let (endpoint, _) = serve_routes(hf_routes("", &[("other.gguf", b"o", b"o")]), 1);
    let err = pull(&m, "hf:org/repo/f.gguf", &root, &endpoint).unwrap_err();
    assert!(
        err.to_string().contains("f.gguf not found in org/repo"),
        "{err}"
    );

    let files = [("f.gguf", b"expected".as_slice(), b"tampered".as_slice())];
    let (endpoint, _) = serve_routes(hf_routes("", &files), 2);
    let err = pull(&m, "hf:org/repo/f.gguf", &root, &endpoint).unwrap_err();
    assert!(err.to_string().contains("SHA-256 mismatch"), "{err}");
    assert!(!root.join("hf/org/repo/f.gguf").exists());

    let (endpoint, _) = serve_routes(Vec::new(), 1);
    let err = pull(&m, "hf:org/repo/f.gguf@dev", &root, &endpoint).unwrap_err();
    assert!(err.to_string().contains("HTTP 404"), "{err}");
    let err = pull(&m, "hf:org/repo/f.gguf", &root, NO_SERVER).unwrap_err();
    assert!(err.to_string().contains("cannot list"), "{err}");
}

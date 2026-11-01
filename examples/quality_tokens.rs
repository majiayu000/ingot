//! CPU-only preparation/decoding using the exact production model tokenizer.
use ingot::{Result, gguf::Gguf, invalid, tokenizer::Tokenizer};
use std::{fs, io::Write, path::Path};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 2 {
        return Err(invalid("quality_tokens MODEL INDEX_TSV"));
    }
    let g = Gguf::open(Path::new(&a[0]))?;
    let tok = Tokenizer::from_gguf(&g)?;
    for line in fs::read_to_string(&a[1])?.lines() {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 3 {
            return Err(invalid("expected mode, input file, output file"));
        }
        let input = fs::read(f[1])?;
        let bytes = match f[0] {
            "chat" | "encode" => {
                let text = std::str::from_utf8(&input).map_err(invalid)?;
                let ids = if f[0] == "chat" {
                    tok.chat(text)?
                } else {
                    tok.encode(text, false)?
                };
                ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()
            }
            "decode" => {
                if input.len() % 4 != 0 {
                    return Err(invalid("partial token"));
                }
                let ids: Vec<_> = input
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                tok.decode(&ids)?.into_bytes()
            }
            _ => return Err(invalid("unknown token preparation mode")),
        };
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(f[2])?
            .write_all(&bytes)?;
        println!("{}\t{}\t{}", f[0], f[2], bytes.len());
    }
    Ok(())
}

use std::io::{Read, self};

mod lisp;

use lisp::{Env, Stream};

fn repl<S: Iterator<Item = char>>(
    stream: &mut Stream<S>,
    env: Env,
) -> io::Result<Env> {
    use lisp::ParseError::*;

    let mut e = env;
    loop {
        let expr = match stream.read_value() {
            Ok(v) => v,
            Err(Eof) => break,
            Err(Lisp(e)) => {
                println!("error: lisp: {e}");
                continue;
            }
        };

        let ast = match expr.build_ast() {
            Ok(a) => a,
            Err(l) => {
                println!("error: lisp: {l}");
                continue;
            },
        };

        let (result, env_prime) = match ast.eval(e.clone()) {
            Ok(x) => x,
            Err(l) => {
                println!("error: lisp: {l}");
                continue;
            },
        };
        e = env_prime;

        println!("{result}");
    }
    Ok(e)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustyline::error::ReadlineError;
    use rustyline::DefaultEditor;

    let mut env = match Env::with_stdlib() {
        Ok(env) => env,
        Err(e) => {
            println!("{e}");
            return Ok(());
        },
    };

    let args: Vec<String> = std::env::args().collect();
    if let Some(path) = args.get(1) {
        let p = std::path::Path::new(path);
        let mut f = std::fs::File::open(p)?;

        let mut buf = Vec::new();
        let _ = f.read_to_end(&mut buf)?;

        let s = String::from_utf8(buf).unwrap_or_else(|_| panic!("bad..."));
        let mut stream = lisp::Stream::from_str(s.as_str());
        _ = repl(&mut stream, env)?;
        return Ok(());
    }

    let mut rl = DefaultEditor::new()?;
    loop {
        let readline = rl.readline("$> ");
        match readline {
            Ok(line) => {
                let mut stream = lisp::Stream::from_str(line.as_str());
                env = repl(&mut stream, env)?;
            },
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    Ok(())
}


use std::io::{Write, Read, self};

mod lisp;
mod vm;

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
    use vm::{Instruction, Procedure};

    let mut program = Vec::new();
    program.push(Instruction::new(Procedure::Push, 4));
    program.push(Instruction::new(Procedure::Push, 1));
    program.push(Instruction::new(Procedure::Push, 11));
    // [4 1 11]

    program.push(Instruction::new(Procedure::Add, 0));
    // [4 12]

    program.push(Instruction::new(Procedure::Mult, 0));
    // [48]

    program.push(Instruction::new(Procedure::Push, 4));
    program.push(Instruction::new(Procedure::Sub, 0));
    // [44]

    program.push(Instruction::new(Procedure::Push, 4));
    program.push(Instruction::new(Procedure::Div, 0));
    // [11]

    let f = std::fs::File::create("program.vm")?;
    vm::save_program(f, program.as_slice())?;

    // TIME TO LOAD

    const MEM_CAPACITY: usize = 64 * 1024;
    let mut stack = [0u8; MEM_CAPACITY];
    let mut program: [vm::Instruction; MEM_CAPACITY] = unsafe { std::mem::zeroed() };

    let f = std::fs::File::open("program.vm")?;
    let mut m = vm::Machine::from_reader(f, &mut stack, &mut program)?;

    m.run()?;
    let res = m.last_value().unwrap();
    println!("res: {res}");

    Ok(())
}

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     use rustyline::error::ReadlineError;
//     use rustyline::DefaultEditor;
//
//     let mut env = match Env::with_stdlib() {
//         Ok(env) => env,
//         Err(e) => {
//             println!("{e}");
//             return Ok(());
//         },
//     };
//
//     let args: Vec<String> = std::env::args().collect();
//     if let Some(path) = args.get(1) {
//         let p = std::path::Path::new(path);
//         let mut f = std::fs::File::open(p)?;
//
//         let mut buf = Vec::new();
//         let _ = f.read_to_end(&mut buf)?;
//
//         let s = String::from_utf8(buf).unwrap_or_else(|_| panic!("bad..."));
//         let mut stream = lisp::Stream::from_str(s.as_str());
//         _ = repl(&mut stream, env)?;
//         return Ok(());
//     }
//
//     let mut rl = DefaultEditor::new()?;
//     loop {
//         let readline = rl.readline("$> ");
//         match readline {
//             Ok(line) => {
//                 let mut stream = lisp::Stream::from_str(line.as_str());
//                 env = repl(&mut stream, env)?;
//             },
//             Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
//             Err(err) => {
//                 println!("Error: {:?}", err);
//                 break;
//             }
//         }
//     }
//     Ok(())
// }
//

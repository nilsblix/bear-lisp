use std::fmt;
use std::io;
use std::error;

use crate::vm::internals::{Value, Instruction, Procedure, ProcedureDefinition, CastError};

const HEADER: [u8; 12] = *b"VM/MAR_26/LE";

pub fn save_program<W: io::Write>(mut w: W, program: &[Instruction]) -> io::Result<()> {
    w.write_all(&HEADER)?;
    let count: u32 = program.len().try_into().map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "program too large")
    })?;
    w.write_all(&count.to_le_bytes())?;

    for ins in program {
        // 9 bytes per instruction
        w.write_all(&[ins.proc as u8])?;
        w.write_all(&ins.operand.to_le_bytes())?;
    }

    Ok(())
}

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    CastError(CastError),
    FaultyHeader,
    BackingTooSmall,
    Truncated,
    UnknownProcedure,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "{}", e),
            LoadError::CastError(e) => write!(f, "{}", e),
            LoadError::FaultyHeader => write!(f, "faulty header"),
            LoadError::BackingTooSmall => write!(f, "program's backing buffer is too small"),
            LoadError::Truncated => write!(f, "instruction got cut of by eof"),
            LoadError::UnknownProcedure => write!(f, "unknown procedure-byte"),
        }
    }
}

impl error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self { LoadError::Io(e) }
}

pub(super) fn load_program<R: io::Read>(mut r: R, out: &mut [Instruction]) -> Result<usize, LoadError> {
    let mut  header = [0u8; HEADER.len()];
    r.read_exact(&mut header)?;
    if header != HEADER {
        return Err(LoadError::FaultyHeader);
    }

    let mut count_bytes = [0u8; 4];
    r.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes) as usize;

    if count > out.len() {
        return Err(LoadError::BackingTooSmall);
    }

    for i in 0..count {
        let mut proc_b = [0u8; 1];
        let mut oper_b = [0u8; 8];

        r.read_exact(&mut proc_b)
            .map_err(|e| if e.kind() == io::ErrorKind::UnexpectedEof {
                LoadError::Truncated
            } else { LoadError::Io(e) })?;

        r.read_exact(&mut oper_b)
            .map_err(|e| if e.kind() == io::ErrorKind::UnexpectedEof {
                LoadError::Truncated
            } else { LoadError::Io(e) })?;

        let proc = Procedure::decode(proc_b[0]);
        if proc.is_none() {
            return Err(LoadError::UnknownProcedure);
        }
        let proc = proc.unwrap();

        out[i] = Instruction::new(proc, Value::from_le_bytes(oper_b));
    }

    Ok(count)
}

pub struct Assembler {
    line_no: u32,
}

#[derive(Debug, PartialEq)]
pub struct AsmError {
    pub row: u32,
    pub msg: String,
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.row, self.msg)
    }
}

impl Assembler {
    pub fn new() -> Self {
        Self{ line_no: 0 }
    }

    pub fn assemble_str(&mut self, source: &str) -> Result<Vec<Instruction>, AsmError> {
        self.assemble(std::io::Cursor::new(source))
    }

    pub fn assemble<R: io::BufRead>(&mut self, reader: R) -> Result<Vec<Instruction>, AsmError> {
        let mut program = Vec::new();

        for l in reader.lines() {
            self.line_no += 1;

            let l = l.map_err(|e| self.err(format!("io error: {}", e)))?;
            let mut split = l.split_whitespace();

            let proc_str = match split.next() {
                Some(p) if p == ";" => continue,
                Some(p) => p,
                None => continue, // empty line
            };

            let proc_def = ProcedureDefinition::decode_str(proc_str)
                .ok_or(self.err(format!("unknown procedure: '{proc_str}'")))?;

            let op_tok = match split.next() {
                Some(";") | None => None,
                Some(v) => Some(v),
            };
            let (op_str, op_val) = match op_tok {
                Some(v) => (v, Value::from_str_radix(v, 10).ok()),
                None => ("", None),
            };

            let op = match (proc_def.expects_operand, op_tok, op_val) {
                (true, _, Some(i)) => i,
                (true, _, None) => {
                    let msg = format!(
                        "procedure '{}' expects an integer operand, found: '{}'",
                        proc_str, op_str
                    );
                    return Err(self.err(msg));
                }
                (false, Some(v), _) => {
                    let msg = format!("procedure '{}' expects no operand, found: '{}'", proc_str, v);
                    return Err(self.err(msg));
                }
                (false, None, _) => 0,
            };

            let ins = Instruction::new(proc_def.proc, op);
            program.push(ins);
        }

        Ok(program)
    }

    fn err(&self, msg: String) -> AsmError {
        AsmError{ row: self.line_no, msg }
    }
}

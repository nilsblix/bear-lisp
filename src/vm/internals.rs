use std::fmt;
use std::error;

pub type Value = i64;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Procedure {
    Nop = 0,
    Push,
    Add,
    Sub,
    Mult,
    Div,
    Jump,
    JumpIfNonZero,
    JumpIfLt,
    JumpIfLe,
    JumpIfGt,
    JumpIfGe,
    Dup,
    Swap,
    Over,
}

pub struct ProcedureDefinition {
    pub proc: Procedure,
    pub name: &'static str,
    pub expects_operand: bool,
}

impl ProcedureDefinition {
    pub fn decode_str(s: &str) -> Option<Self> {
        for def in PROC_DEFS {
            if s == def.name {
                return Some(def);
            }
        }
        None
    }
}

const PROC_DEFS: [ProcedureDefinition; 15] = [
    ProcedureDefinition {
        proc: Procedure::Nop,
        name: "nop",
        expects_operand: false,
    },
    ProcedureDefinition {
        proc: Procedure::Push,
        name: "push",
        expects_operand: true,
    },
    ProcedureDefinition {
        proc: Procedure::Add,
        name: "add",
        expects_operand: false,
    },
    ProcedureDefinition {
        proc: Procedure::Sub,
        name: "sub",
        expects_operand: false,
    },
    ProcedureDefinition {
        proc: Procedure::Mult,
        name: "mult",
        expects_operand: false,
    },
    ProcedureDefinition {
        proc: Procedure::Div,
        name: "div",
        expects_operand: false,
    },
    ProcedureDefinition {
        proc: Procedure::Jump,
        name: "jmp",
        expects_operand: true,
    },
    ProcedureDefinition {
        proc: Procedure::JumpIfNonZero,
        name: "jnz",
        expects_operand: true,
    },
    ProcedureDefinition {
        proc: Procedure::JumpIfLt,
        name: "jlt",
        expects_operand: true,
    },
    ProcedureDefinition {
        proc: Procedure::JumpIfLe,
        name: "jle",
        expects_operand: true,
    },
    ProcedureDefinition {
        proc: Procedure::JumpIfGt,
        name: "jgt",
        expects_operand: true,
    },
    ProcedureDefinition {
        proc: Procedure::JumpIfGe,
        name: "jge",
        expects_operand: true,
    },
    ProcedureDefinition {
        proc: Procedure::Dup,
        name: "dup",
        expects_operand: false,
    },
    ProcedureDefinition {
        proc: Procedure::Swap,
        name: "swap",
        expects_operand: false,
    },
    ProcedureDefinition {
        proc: Procedure::Over,
        name: "over",
        expects_operand: false,
    },
];

impl Procedure {
    pub fn decode(b: u8) -> Option<Self> {
        const LEN: u8 = if PROC_DEFS.len() > u8::max_value() as usize {
            u8::max_value()
        } else {
            PROC_DEFS.len() as u8
        };
        match b {
            0..LEN => Some(PROC_DEFS[b as usize].proc),
            _ => None,
        }
    }
}

/// NOTE We don't make some Procedures carry an operand because in the future we
/// want to try to implement a #![no_std] version of this vm. Therefore if we
/// zero out some instruction space (future implementation of vm's debug mode
/// with a static stack size), all instructions automatically get set to Nop,
/// which catches runtime bugs and makes the machine panic, instead of producing
/// weird side-effects.
///
/// TODO check if we really need this repr(C) and _pad. rustc might
/// automatically implement it for us.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instruction {
    pub proc: Procedure,
    _pad: [u8; 7],
    /// Operand is ignored for some operands. Errors inside Assembler or simply
    /// ignored when cast from bytecode.
    pub operand: Value,
}

impl Instruction {
    pub fn new(proc: Procedure, operand: Value) -> Self {
        Self {
            proc,
            _pad: [0u8; 7],
            operand,
        }
    }
}

#[derive(Debug)]
pub enum CastError {
    IndivisibleSize,
    Unaligned,
    UncastableType,
}

impl fmt::Display for CastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CastError::IndivisibleSize => write!(f, "indivisible size"),
            CastError::Unaligned => write!(f, "unaligned"),
            CastError::UncastableType => write!(f, "tried to cast to type with no size"),
        }
    }
}

impl error::Error for CastError {}

fn check_castable<F, T>(source: &[F]) -> Result<usize, CastError> {
    use std::mem::{size_of, align_of};
    if size_of::<T>() == 0 {
        return Err(CastError::UncastableType);
    }

    if source.len() % size_of::<T>() != 0 {
        return Err(CastError::IndivisibleSize);
    }

    let ptr = source.as_ptr();
    if ptr.align_offset(align_of::<T>()) != 0 {
        return Err(CastError::Unaligned);
    }

    Ok(source.len() / std::mem::size_of::<T>())
}

pub(super) fn cast_slice_mut<F, T>(source: &mut [F]) -> Result<&mut [T], CastError> {
    let len = check_castable::<F, T>(source)?;
    let ptr = source.as_mut_ptr();
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr as *mut T, len) })
}

use std::fmt;
use std::error;

pub type Value = i64;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opcode {
    Nop,
    Push,
    True,
    False,
    Pop,
    Dup,
    Swap,
    Over,

    Add,
    Sub,
    Mult,
    Div,

    NumEq,
    LT,
    GT,

    Jump,
    JumpIfNonZero,

    // Return,
    //
    // TODO Implement these. Not sure about all of them, but most should come in
    // handy when compiling the lisp.
    // LoadLocal,
    // StoreLocal,
    // LoadGlobal,
    // StoreGlobal,
    // DefineGlobal,
    //
    // Call,
    // Closure,
    // LoadUpValue,
    // StoreUpValue,
    //
    // Cons,
    // Car,
    // Cdr,
    //
    // IsNil,
    // IsPair,
    // IsSymbol,
}

pub struct OpcodeDef {
    pub op: Opcode,
    pub name: &'static str,
    pub expects_operand: bool,
}

impl OpcodeDef {
    pub fn decode_str(s: &str) -> Option<Self> {
        for def in OPCODE_DEFS {
            if s == def.name {
                return Some(def);
            }
        }
        None
    }
}

const OPCODE_DEFS: [OpcodeDef; 17] = [
    OpcodeDef {
        op: Opcode::Nop,
        name: "nop",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Push,
        name: "push",
        expects_operand: true,
    },
    OpcodeDef {
        op: Opcode::True,
        name: "true",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::False,
        name: "false",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Pop,
        name: "pop",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Dup,
        name: "dup",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Swap,
        name: "swap",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Over,
        name: "over",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Add,
        name: "add",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Sub,
        name: "sub",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Mult,
        name: "mult",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Div,
        name: "div",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::NumEq,
        name: "numeq",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::LT,
        name: "lt",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::GT,
        name: "gt",
        expects_operand: false,
    },
    OpcodeDef {
        op: Opcode::Jump,
        name: "jump",
        expects_operand: true,
    },
    OpcodeDef {
        op: Opcode::JumpIfNonZero,
        name: "jump_on_nz",
        expects_operand: true,
    },
];

impl Opcode {
    pub fn decode(b: u8) -> Option<Self> {
        const LEN: u8 = if OPCODE_DEFS.len() > u8::max_value() as usize {
            u8::max_value()
        } else {
            OPCODE_DEFS.len() as u8
        };
        match b {
            0..LEN => Some(OPCODE_DEFS[b as usize].op),
            _ => None,
        }
    }
}

/// NOTE We don't make some Operations carry an operand because in the future we
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
    pub op: Opcode,
    _pad: [u8; 7],
    /// Operand is ignored for some operands. Errors inside Assembler or simply
    /// ignored when cast from bytecode.
    pub operand: Value,
}

impl Instruction {
    pub fn new(opcode: Opcode, operand: Value) -> Self {
        Self {
            op: opcode,
            _pad: [0u8; 7],
            operand,
        }
    }

    pub fn zeroed() -> Self {
        Self::new(Opcode::Nop, 0)
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

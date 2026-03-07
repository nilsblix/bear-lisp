use std::fmt;
use std::io;
use std::error;

use crate::vm::internals::{Value, Instruction, Opcode, self};
use crate::vm::asm::{Assembler, LoadError, self};

#[derive(Debug)]
pub enum Error {
    StackOverflow,
    StackUnderflow,
    DivByZero,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::StackOverflow => write!(f, "stack overflow"),
            Error::StackUnderflow => write!(f, "stack underflow"),
            Error::DivByZero => write!(f, "tried to divide by zero"),
        }
    }
}

impl error::Error for Error {}


pub struct Machine<'m> {
    pub stack: &'m mut [Value],
    pub sp: usize,
    pub program: &'m [Instruction],
    pub ip: usize,
    // TODO Ram/heap. Might look something like this to make StackValue be an
    // enum over i64 and u32, where HeapValue contains some metadata, ex some
    // header with type-info.
    // mem: &'m mut [HeapValue],
}

impl<'a> Machine<'a> {
    pub fn new(stack: &'a mut [Value], program: &'a [Instruction]) -> Self {
        Self { stack, sp: 0, program, ip: 0 }
    }

    pub fn from_ir(stack: &'a mut [u8], program: &'a [Instruction]) -> Result<Self, internals::CastError> {
        let stack = internals::cast_slice_mut::<u8, Value>(stack)?;
        Ok(Self::new(stack, program))
    }

    pub fn from_reader<R: io::Read>(r: R, stack: &'a mut [u8], program: &'a mut [Instruction]) -> Result<Self, LoadError> {
        let n = asm::load_program(r, program)?;
        let ir = &program[0..n];
        Self::from_ir(stack, ir).map_err(|e| LoadError::CastError(e))
    }

    fn next_instruction(&mut self) -> Option<Instruction> {
        if self.ip >= self.program.len() {
            None
        } else {
            let popped = self.program[self.ip];
            self.ip += 1;
            Some(popped)
        }
    }

    fn push_stack(&mut self, v: Value) -> Result<(), Error> {
        if self.sp >= self.stack.len() {
            Err(Error::StackOverflow)
        } else {
            self.stack[self.sp] = v;
            self.sp += 1;
            Ok(())
        }
    }

    fn pop_stack(&mut self) -> Result<Value, Error> {
        if self.sp == 0 {
            Err(Error::StackUnderflow)
        } else {
            self.sp -= 1;
            Ok(self.stack[self.sp])
        }
    }

    pub fn last_value(&self) -> Option<Value> {
        if self.sp == 0 {
            None
        } else {
            Some(self.stack[self.sp - 1])
        }
    }

    /// The machine halts when `ip` reaches the end of the program.
    pub fn run(&mut self) -> Result<(), Error> {
        use Opcode::*;
        loop {
            let ins = match self.next_instruction() {
                Some(i) => i,
                None => return Ok(()),
            };

            macro_rules! binary_op {
                ($op:tt) => {
                    {
                        let b = self.pop_stack()?;
                        let a = self.pop_stack()?;
                        self.push_stack(a $op b)?;
                    }
                };
            }

            macro_rules! binary_cmp {
                ($op:tt) => {
                    {
                        let rhs = self.pop_stack()?;
                        let lhs = self.pop_stack()?;
                        if lhs $op rhs {
                            self.push_stack(1)?;
                        } else {
                            self.push_stack(0)?;
                        }
                    }
                };
            }

            match ins.op {
                Nop => continue,
                Push => self.push_stack(ins.operand)?,
                True => self.push_stack(1)?,
                False => self.push_stack(0)?,
                Pop => _ = self.pop_stack()?,
                Dup => self.push_stack(self.last_value().ok_or(Error::StackUnderflow)?)?,
                Swap => {
                    let b = self.pop_stack()?;
                    let a = self.pop_stack()?;
                    self.push_stack(b)?;
                    self.push_stack(a)?;
                },
                Over => {
                    if self.sp < 2 {
                        return Err(Error::StackUnderflow);
                    }
                    let v = self.stack[self.sp - 2];
                    self.push_stack(v)?;
                }

                Add => binary_op!(+),
                Sub => binary_op!(-),
                Mult => binary_op!(*),
                Div => {
                    let b = self.pop_stack()?;
                    if b == 0 {
                        return Err(Error::DivByZero);
                    }
                   let a = self.pop_stack()?;
                    self.push_stack(a / b)?;
                },

                NumEq => binary_cmp!(==),
                LT => binary_cmp!(<),
                GT => binary_cmp!(>),

                Jump => self.ip = ins.operand as usize,
                JumpIfNonZero => {
                    if self.pop_stack()? != 0 {
                        self.ip = ins.operand as usize;
                    }
                },
            }
        }
    }

    /// Helper method to help discoverability of Assembler
    #[allow(dead_code)]
    pub fn assembler() -> Assembler {
        Assembler::new()
    }
}

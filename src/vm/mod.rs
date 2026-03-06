mod internals;
pub mod asm;
pub mod machine;

// load gets called from Machine::from_xxx because load uses casting, which is
// considered unsafe, and therefore not exposed to the world.
pub use asm::save_program;
pub use internals::Instruction;
pub use machine::Machine;

#[cfg(test)]
mod tests {
    const MEM_CAPACITY: usize = 64 * 1024;
    use crate::vm::internals::{Instruction, Opcode};
    use crate::vm::asm::{Assembler, AsmError};
    use crate::vm::machine::Machine;

    #[test]
    fn assemble() {
        let source = "
            push 4
            push 1
            push 11  ; [4 1 11]
            add      ; [4 12]
            mult     ; [48]
            push 4
            sub      ; [44]
            push 4
            div      ; [11]
            ";
        let mut asm = Assembler::new();
        let res = asm.assemble_str(source);
        if let Err(e) = &res {
            println!("error ({}): {}", e.row, e.msg);
            assert!(false);
        }

        let parsed = res.ok().unwrap();

        let mut program = Vec::new();
        program.push(Instruction::new(Opcode::Push, 4));
        program.push(Instruction::new(Opcode::Push, 1));
        program.push(Instruction::new(Opcode::Push, 11));
        program.push(Instruction::new(Opcode::Add, 0));
        program.push(Instruction::new(Opcode::Mult, 0));
        program.push(Instruction::new(Opcode::Push, 4));
        program.push(Instruction::new(Opcode::Sub, 0));
        program.push(Instruction::new(Opcode::Push, 4));
        program.push(Instruction::new(Opcode::Div, 0));

        assert_eq!(parsed.len(), program.len());
        for (got, exp) in parsed.iter().zip(program) {
            assert_eq!(got.clone(), exp);
        }

        let source = "
            push 4
            push     ; sdlkfjsldkfj
            push 11  ; [4 1 11]
            add  1   ; [4 12]
            mult     ; [48]
            push 4
            sub      ; [44]
            push 4
            div      ; [11]
            ";
        let mut asm = Assembler::new();
        let res = asm.assemble_str(source);
        assert_eq!(res, Err(AsmError{ row: 3, msg: "operation 'push' expects an operand, found: ''".to_string() }));

        let source = "
            push 4
            push 1
            push 11  ; [4 1 11]
            add  1   ; [4 12]
            mult     ; [48]
            push 4
            sub      ; [44]
            push 4
            div      ; [11]
            ";
        let mut asm = Assembler::new();
        let res = asm.assemble_str(source);
        assert_eq!(res, Err(AsmError{ row: 5, msg: "operation 'add' doesn't expect an operand, found: '1'".to_string() }));

        let source = "
            push 4
            push 1
            push 11  ; [4 1 11]
            add  a   ; [4 12]
            mult     ; [48]
            push 4
            sub      ; [44]
            push 4
            div      ; [11]
            ";
        let mut asm = Assembler::new();
        let res = asm.assemble_str(source);
        assert_eq!(res, Err(AsmError{ row: 5, msg: "operation 'add' doesn't expect an operand, found: 'a'".to_string() }));
    }

    #[test]
    fn simple_run() {
        let mut program = Vec::new();

        program.push(Instruction::new(Opcode::Push, 4));
        program.push(Instruction::new(Opcode::Push, 1));
        program.push(Instruction::new(Opcode::Push, 11));
        // [4 1 11]

        program.push(Instruction::new(Opcode::Add, 0));
        // [4 12]

        program.push(Instruction::new(Opcode::Mult, 0));
        // [48]

        program.push(Instruction::new(Opcode::Push, 4));
        program.push(Instruction::new(Opcode::Sub, 0));
        // [44]

        program.push(Instruction::new(Opcode::Push, 4));
        program.push(Instruction::new(Opcode::Div, 0));
        // [11]

        let mut stack = [0u8; MEM_CAPACITY];

        let mut m = Machine::from_ir(&mut stack, program.as_slice()).unwrap();
        assert_eq!(m.head, 0);
        assert_eq!(m.ip, 0);

        m.run().unwrap();
        let res = m.last_value().unwrap();
        assert_eq!(res, 11);
    }

    #[test]
    fn simple_fibonacci() {
        let mut program = Vec::new();
        program.push(Instruction::new(Opcode::Push, 0)); // f(0)
        program.push(Instruction::new(Opcode::Push, 1)); // f(1)

        for _ in 0..9 {
            program.push(Instruction::new(Opcode::Swap, 0));
            program.push(Instruction::new(Opcode::Over, 0));
            program.push(Instruction::new(Opcode::Add, 0));
        }

        let mut stack = [0u8; MEM_CAPACITY];
        let mut m = Machine::from_ir(&mut stack, program.as_slice()).unwrap();
        m.run().unwrap();

        assert_eq!(m.head, 2);
        assert_eq!(m.stack[m.head - 2], 34);
        assert_eq!(m.last_value(), Some(55));
    }
}

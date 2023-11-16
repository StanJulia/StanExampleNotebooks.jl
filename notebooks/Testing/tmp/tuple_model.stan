parameters {
    real r;
    matrix[2, 3] x;
    tuple(real, real) bar;
    tuple(real, tuple(real, real), real) bar2;
    tuple(real, tuple(real, tuple(real, real)), real) bar3;
}
model {
    r ~ std_normal();

    for (i in 1:2) {
        x[i,:] ~ std_normal();
    }

    bar.1 ~ std_normal();
    bar.2 ~ std_normal();
    bar2.1 ~ std_normal();
    bar2.2.1 ~ std_normal();
    bar2.2.2 ~ std_normal();
	bar2.3 ~ std_normal();
    bar3.1 ~ std_normal();
    bar3.2.1 ~ std_normal();
    bar3.2.2.1 ~ std_normal();
    bar3.2.2.2 ~ std_normal();
	bar3.3 ~ std_normal();
}
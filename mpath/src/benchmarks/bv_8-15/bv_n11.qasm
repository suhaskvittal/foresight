OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[10];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
x q[10];
h q[10];
cx q[0], q[10];
cx q[1], q[10];
cx q[2], q[10];
cx q[3], q[10];
cx q[5], q[10];
cx q[6], q[10];
cx q[8], q[10];
cx q[9], q[10];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];

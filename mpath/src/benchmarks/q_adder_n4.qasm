OPENQASM 2.0;
include "qelib1.inc";

qreg	q[4];
creg	c[4];

// q0 = A, q1 = B, q2 = Cin/S, q3 = Cout
ccx	q[0],q[1],q[3];
cx	q[0],q[1];
ccx q[1],q[2],q[3];
cx	q[1],q[2];
cx	q[0],q[1];  // Restore B.
barrier q;
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];

OPENQASM 2.0;
include "qelib1.inc";
qreg q6[4];
h q6[3];
h q6[2];
h q6[1];
h q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
h q6[3];
h q6[2];
h q6[1];
h q6[0];
rx(-pi/2) q6[3];
rx(-pi/2) q6[2];
rx(-pi/2) q6[1];
rx(-pi/2) q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
rx(pi/2) q6[3];
rx(pi/2) q6[2];
rx(pi/2) q6[1];
rx(pi/2) q6[0];
h q6[3];
rx(-pi/2) q6[2];
h q6[1];
rx(-pi/2) q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
h q6[3];
rx(pi/2) q6[2];
h q6[1];
rx(pi/2) q6[0];
rx(-pi/2) q6[3];
h q6[2];
rx(-pi/2) q6[1];
h q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
rx(pi/2) q6[3];
h q6[2];
rx(pi/2) q6[1];
h q6[0];
rx(-pi/2) q6[3];
rx(-pi/2) q6[2];
h q6[1];
h q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
rx(pi/2) q6[3];
rx(pi/2) q6[2];
h q6[1];
h q6[0];
h q6[3];
h q6[2];
rx(-pi/2) q6[1];
rx(-pi/2) q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
h q6[3];
h q6[2];
rx(pi/2) q6[1];
rx(pi/2) q6[0];
rx(-pi/2) q6[3];
h q6[2];
h q6[1];
rx(-pi/2) q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
rx(pi/2) q6[3];
h q6[2];
h q6[1];
rx(pi/2) q6[0];
h q6[3];
rx(-pi/2) q6[2];
rx(-pi/2) q6[1];
h q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(0.90269802) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
h q6[3];
rx(pi/2) q6[2];
rx(pi/2) q6[1];
h q6[0];
h q6[1];
h q6[0];
cx q6[1],q6[0];
rz(-2.6322493) q6[0];
cx q6[1],q6[0];
h q6[1];
h q6[0];
rx(-pi/2) q6[1];
rx(-pi/2) q6[0];
cx q6[1],q6[0];
rz(-2.6322493) q6[0];
cx q6[1],q6[0];
rx(-pi/2) q6[1];
rx(-pi/2) q6[0];
h q6[2];
h q6[0];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(-1.3884772) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
h q6[2];
h q6[0];
rx(-pi/2) q6[2];
rx(-pi/2) q6[0];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(-1.3884772) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
rx(-pi/2) q6[2];
rx(-pi/2) q6[0];
h q6[2];
h q6[1];
cx q6[2],q6[1];
rz(-1.5096491) q6[1];
cx q6[2],q6[1];
h q6[2];
h q6[1];
rx(-pi/2) q6[2];
rx(-pi/2) q6[1];
cx q6[2],q6[1];
rz(-1.5096491) q6[1];
cx q6[2],q6[1];
rx(-pi/2) q6[2];
rx(-pi/2) q6[1];
h q6[3];
h q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(2.2701798) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
h q6[3];
h q6[0];
rx(-pi/2) q6[3];
rx(-pi/2) q6[0];
cx q6[3],q6[2];
cx q6[2],q6[1];
cx q6[1],q6[0];
rz(2.2701798) q6[0];
cx q6[1],q6[0];
cx q6[2],q6[1];
cx q6[3],q6[2];
rx(-pi/2) q6[3];
rx(-pi/2) q6[0];
h q6[3];
h q6[1];
cx q6[3],q6[2];
cx q6[2],q6[1];
rz(-0.758358) q6[1];
cx q6[2],q6[1];
cx q6[3],q6[2];
h q6[3];
h q6[1];
rx(-pi/2) q6[3];
rx(-pi/2) q6[1];
cx q6[3],q6[2];
cx q6[2],q6[1];
rz(-0.758358) q6[1];
cx q6[2],q6[1];
cx q6[3],q6[2];
rx(-pi/2) q6[3];
rx(-pi/2) q6[1];
h q6[3];
h q6[2];
cx q6[3],q6[2];
rz(-1.8998198) q6[2];
cx q6[3],q6[2];
h q6[3];
h q6[2];
rx(-pi/2) q6[3];
rx(-pi/2) q6[2];
cx q6[3],q6[2];
rz(-1.8998198) q6[2];
cx q6[3],q6[2];
rx(-pi/2) q6[3];
rx(-pi/2) q6[2];
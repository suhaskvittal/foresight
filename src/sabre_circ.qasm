OPENQASM 2.0;
include "qelib1.inc";
qreg q[53];
creg c[13];
u3(pi/4,pi/2,0) q[21];
u3(pi/4,pi/2,0) q[22];
u3(pi/4,pi/2,0) q[23];
u3(pi/4,pi/2,0) q[24];
u3(pi/2,-pi,-pi/4) q[29];
u3(pi/2,0,0) q[30];
u3(pi/2,0,pi/4) q[31];
cx q[30],q[31];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[31];
cx q[30],q[31];
u3(0,-pi,pi/2) q[30];
u3(0,-3*pi/4,-3*pi/4) q[30];
u3(pi/4,-pi/2,pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
u3(pi/4,pi/2,0) q[32];
u3(pi/4,pi/2,0) q[33];
u3(pi/2,-pi,-pi/4) q[34];
u3(pi/4,pi/2,0) q[38];
cx q[30],q[38];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[38];
cx q[30],q[38];
u3(pi,pi/2,-pi/2) q[30];
u3(pi,-1.4612288,0.10956749) q[30];
cx q[30],q[29];
u3(1.5454924,1.3467702,-1.7948224) q[29];
u3(pi,-1.1071487,0.46364761) q[30];
cx q[30],q[29];
u3(pi/4,pi/2,pi/2) q[29];
u3(3*pi/4,pi/2,-pi) q[29];
u3(0,-pi/2,pi/2) q[30];
u3(pi,-1.4612288,0.10956749) q[30];
swap q[30],q[31];
u3(pi/4,0,pi/2) q[38];
u3(pi/4,-pi/2,-pi) q[38];
cx q[30],q[38];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[38];
cx q[30],q[38];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
cx q[30],q[29];
u3(1.5454924,1.3467702,-1.7948224) q[29];
u3(pi,-1.1071487,0.46364761) q[30];
cx q[30],q[29];
u3(pi/4,0,pi/2) q[29];
u3(pi/4,-pi/2,-pi) q[29];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(3*pi/4,0,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
swap q[30],q[38];
cx q[30],q[29];
u3(1.5454924,1.3467702,-1.7948224) q[29];
u3(pi,-1.1071487,0.46364761) q[30];
cx q[30],q[29];
u3(3*pi/4,0,-pi/2) q[29];
u3(0,-pi,-pi/2) q[29];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi/2,-pi,-pi/4) q[39];
cx q[31],q[39];
u3(pi,-1.1071487,0.46364761) q[31];
u3(1.5454924,1.3467702,-1.7948224) q[39];
cx q[31],q[39];
u3(0,-pi/2,pi/2) q[31];
u3(0,-3*pi/4,-3*pi/4) q[31];
cx q[31],q[32];
u3(pi,-1.1071487,0.46364761) q[31];
u3(1.5454924,1.3467702,-1.7948224) q[32];
cx q[31],q[32];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-3*pi/4,-3*pi/4) q[31];
cx q[31],q[21];
u3(1.5454924,1.3467702,-1.7948224) q[21];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[21];
u3(pi/4,0,pi/2) q[21];
u3(3*pi/4,pi/2,-pi) q[21];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-3*pi/4,-3*pi/4) q[31];
u3(pi/4,0,pi/2) q[32];
u3(pi/4,-pi/2,-pi) q[32];
swap q[31],q[32];
cx q[32],q[22];
u3(1.5454924,1.3467702,-1.7948224) q[22];
u3(pi,-1.1071487,0.46364761) q[32];
cx q[32],q[22];
u3(pi/4,0,pi/2) q[22];
u3(pi/4,-pi/2,-pi) q[22];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-3*pi/4,-3*pi/4) q[32];
u3(pi/4,pi/2,pi/2) q[39];
u3(pi/4,-pi/2,-pi) q[39];
cx q[38],q[39];
u3(pi,-1.1071487,0.46364761) q[38];
u3(1.5454924,1.3467702,-1.7948224) q[39];
cx q[38],q[39];
u3(pi,pi/2,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
u3(3*pi/4,0,-pi/2) q[39];
u3(3*pi/4,pi/2,-pi) q[39];
swap q[38],q[39];
cx q[30],q[38];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[38];
cx q[30],q[38];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi/4,0,pi/2) q[38];
u3(pi/4,-pi/2,-pi) q[38];
cx q[39],q[31];
u3(1.5454924,1.3467702,-1.7948224) q[31];
u3(pi,-1.1071487,0.46364761) q[39];
cx q[39],q[31];
u3(3*pi/4,0,-pi/2) q[31];
u3(pi/4,-pi/2,-pi) q[31];
cx q[30],q[31];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[31];
cx q[30],q[31];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(3*pi/4,0,-pi/2) q[31];
u3(3*pi/4,pi/2,-pi) q[31];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
swap q[39],q[31];
cx q[31],q[21];
u3(1.5454924,1.3467702,-1.7948224) q[21];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[21];
u3(pi/4,0,pi/2) q[21];
u3(3*pi/4,pi/2,-pi) q[21];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
u3(pi/4,pi/2,0) q[40];
cx q[32],q[40];
u3(pi,-1.1071487,0.46364761) q[32];
u3(1.5454924,1.3467702,-1.7948224) q[40];
cx q[32],q[40];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-3*pi/4,-3*pi/4) q[32];
cx q[32],q[33];
u3(pi,-1.1071487,0.46364761) q[32];
u3(1.5454924,1.3467702,-1.7948224) q[33];
cx q[32],q[33];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-3*pi/4,-3*pi/4) q[32];
swap q[22],q[32];
cx q[22],q[23];
u3(pi,-1.1071487,0.46364761) q[22];
u3(1.5454924,1.3467702,-1.7948224) q[23];
cx q[22],q[23];
u3(pi,pi/2,-pi/2) q[22];
u3(0,-3*pi/4,-3*pi/4) q[22];
u3(pi/4,0,pi/2) q[23];
u3(pi/4,-pi/2,-pi) q[23];
swap q[23],q[22];
cx q[23],q[24];
u3(pi,-1.1071487,0.46364761) q[23];
u3(1.5454924,1.3467702,-1.7948224) q[24];
cx q[23],q[24];
u3(pi,pi/2,-pi/2) q[23];
u3(pi,-1.4612288,0.10956749) q[23];
u3(pi/4,0,pi/2) q[24];
u3(3*pi/4,pi/2,-pi) q[24];
swap q[24],q[23];
cx q[24],q[34];
u3(pi,-1.1071487,0.46364761) q[24];
cx q[31],q[32];
u3(pi,-1.1071487,0.46364761) q[31];
u3(1.5454924,1.3467702,-1.7948224) q[32];
cx q[31],q[32];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
u3(3*pi/4,0,-pi/2) q[32];
u3(pi/4,-pi/2,-pi) q[32];
swap q[32],q[31];
swap q[31],q[30];
cx q[31],q[21];
u3(1.5454924,1.3467702,-1.7948224) q[21];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[21];
u3(pi/4,0,pi/2) q[21];
u3(pi/4,-pi/2,-pi) q[21];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
cx q[31],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[30];
u3(3*pi/4,0,-pi/2) q[30];
u3(pi/4,-pi/2,-pi) q[30];
swap q[30],q[29];
cx q[30],q[38];
u3(pi,-1.1071487,0.46364761) q[30];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
u3(pi/4,0,pi/2) q[33];
u3(3*pi/4,pi/2,-pi) q[33];
u3(1.5454924,1.3467702,-1.7948224) q[34];
cx q[24],q[34];
u3(2.3428827,-pi/2,pi/2) q[24];
u3(pi/4,pi/2,pi/2) q[34];
u3(3*pi/4,pi/2,-pi) q[34];
u3(1.5454924,1.3467702,-1.7948224) q[38];
cx q[30],q[38];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(3*pi/4,0,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
swap q[39],q[31];
cx q[30],q[31];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[31];
cx q[30],q[31];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi/4,0,pi/2) q[31];
u3(3*pi/4,pi/2,-pi) q[31];
u3(pi/4,0,pi/2) q[40];
u3(3*pi/4,pi/2,-pi) q[40];
cx q[32],q[40];
u3(pi,-1.1071487,0.46364761) q[32];
u3(1.5454924,1.3467702,-1.7948224) q[40];
cx q[32],q[40];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-pi,-pi/2) q[32];
cx q[32],q[33];
u3(pi,-1.1071487,0.46364761) q[32];
u3(1.5454924,1.3467702,-1.7948224) q[33];
cx q[32],q[33];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-pi,-pi/2) q[32];
cx q[32],q[22];
u3(1.5454924,1.3467702,-1.7948224) q[22];
u3(pi,-1.1071487,0.46364761) q[32];
cx q[32],q[22];
u3(3*pi/4,0,-pi/2) q[22];
u3(3*pi/4,pi/2,-pi) q[22];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-pi,-pi/2) q[32];
u3(pi/4,0,pi/2) q[33];
u3(pi/4,-pi/2,-pi) q[33];
swap q[32],q[33];
cx q[33],q[23];
u3(1.5454924,1.3467702,-1.7948224) q[23];
u3(pi,-1.1071487,0.46364761) q[33];
cx q[33],q[23];
u3(pi/4,0,pi/2) q[23];
u3(pi/4,-pi/2,-pi) q[23];
u3(pi,pi/2,-pi/2) q[33];
u3(0,-pi,-pi/2) q[33];
cx q[33],q[34];
u3(pi,-1.1071487,0.46364761) q[33];
u3(1.5454924,1.3467702,-1.7948224) q[34];
cx q[33],q[34];
u3(0.79871,pi/2,-pi/2) q[33];
u3(pi/4,0,pi/2) q[34];
u3(3*pi/4,pi/2,-pi) q[34];
swap q[33],q[34];
u3(pi/4,0,pi/2) q[40];
u3(3*pi/4,pi/2,-pi) q[40];
cx q[39],q[40];
u3(pi,-1.1071487,0.46364761) q[39];
u3(1.5454924,1.3467702,-1.7948224) q[40];
cx q[39],q[40];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
swap q[31],q[39];
cx q[31],q[32];
u3(pi,-1.1071487,0.46364761) q[31];
u3(1.5454924,1.3467702,-1.7948224) q[32];
cx q[31],q[32];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
swap q[21],q[31];
cx q[21],q[22];
u3(pi,-1.1071487,0.46364761) q[21];
u3(1.5454924,1.3467702,-1.7948224) q[22];
cx q[21],q[22];
u3(pi,pi/2,-pi/2) q[21];
u3(0,-pi,-pi/2) q[21];
u3(pi/4,0,pi/2) q[22];
u3(3*pi/4,pi/2,-pi) q[22];
cx q[30],q[31];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[31];
cx q[30],q[31];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
cx q[30],q[29];
u3(1.5454924,1.3467702,-1.7948224) q[29];
u3(pi,-1.1071487,0.46364761) q[30];
cx q[30],q[29];
u3(3*pi/4,0,-pi/2) q[29];
u3(pi/4,-pi/2,-pi) q[29];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(3*pi/4,0,-pi/2) q[31];
u3(3*pi/4,pi/2,-pi) q[31];
swap q[31],q[30];
u3(3*pi/4,0,-pi/2) q[32];
u3(pi/4,-pi/2,-pi) q[32];
cx q[38],q[39];
u3(pi,-1.1071487,0.46364761) q[38];
u3(1.5454924,1.3467702,-1.7948224) q[39];
cx q[38],q[39];
u3(pi,pi/2,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
cx q[38],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[38];
cx q[38],q[30];
u3(pi/4,0,pi/2) q[30];
u3(pi/4,-pi/2,-pi) q[30];
u3(pi,pi/2,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
swap q[30],q[38];
cx q[30],q[29];
u3(1.5454924,1.3467702,-1.7948224) q[29];
u3(pi,-1.1071487,0.46364761) q[30];
cx q[30],q[29];
u3(3*pi/4,0,-pi/2) q[29];
u3(pi/4,-pi/2,-pi) q[29];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi/4,0,pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
cx q[39],q[38];
u3(1.5454924,1.3467702,-1.7948224) q[38];
u3(pi,-1.1071487,0.46364761) q[39];
cx q[39],q[38];
u3(3*pi/4,0,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
swap q[31],q[39];
swap q[31],q[30];
cx q[30],q[29];
u3(1.5454924,1.3467702,-1.7948224) q[29];
u3(pi,-1.1071487,0.46364761) q[30];
cx q[30],q[29];
u3(3*pi/4,0,-pi/2) q[29];
u3(pi/4,-pi/2,-pi) q[29];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
swap q[38],q[30];
cx q[30],q[29];
u3(1.5454924,1.3467702,-1.7948224) q[29];
u3(pi,-1.1071487,0.46364761) q[30];
cx q[30],q[29];
u3(3*pi/4,0,-pi/2) q[29];
u3(0,-pi,-pi/2) q[29];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi/4,0,pi/2) q[40];
u3(3*pi/4,pi/2,-pi) q[40];
cx q[39],q[40];
u3(pi,-1.1071487,0.46364761) q[39];
u3(1.5454924,1.3467702,-1.7948224) q[40];
cx q[39],q[40];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
u3(pi/4,0,pi/2) q[40];
u3(3*pi/4,pi/2,-pi) q[40];
swap q[39],q[40];
cx q[31],q[39];
u3(pi,-1.1071487,0.46364761) q[31];
u3(1.5454924,1.3467702,-1.7948224) q[39];
cx q[31],q[39];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
u3(pi/4,0,pi/2) q[39];
u3(pi/4,-pi/2,-pi) q[39];
cx q[38],q[39];
u3(pi,-1.1071487,0.46364761) q[38];
u3(1.5454924,1.3467702,-1.7948224) q[39];
cx q[38],q[39];
u3(pi,pi/2,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
u3(3*pi/4,0,-pi/2) q[39];
u3(3*pi/4,pi/2,-pi) q[39];
swap q[38],q[39];
cx q[30],q[38];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[38];
cx q[30],q[38];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi/4,0,pi/2) q[38];
u3(pi/4,-pi/2,-pi) q[38];
swap q[30],q[38];
cx q[29],q[30];
u3(pi,-1.1071487,0.46364761) q[29];
u3(1.5454924,1.3467702,-1.7948224) q[30];
cx q[29],q[30];
u3(pi,pi/2,-pi/2) q[29];
u3(0,-pi,-pi/2) q[29];
swap q[29],q[37];
u3(3*pi/4,0,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
cx q[40],q[32];
u3(1.5454924,1.3467702,-1.7948224) q[32];
u3(pi,-1.1071487,0.46364761) q[40];
cx q[40],q[32];
u3(3*pi/4,0,-pi/2) q[32];
u3(pi/4,-pi/2,-pi) q[32];
cx q[31],q[32];
u3(pi,-1.1071487,0.46364761) q[31];
u3(1.5454924,1.3467702,-1.7948224) q[32];
cx q[31],q[32];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
u3(3*pi/4,0,-pi/2) q[32];
u3(3*pi/4,pi/2,-pi) q[32];
u3(pi,pi/2,-pi/2) q[40];
u3(0,-pi,-pi/2) q[40];
swap q[40],q[32];
cx q[32],q[22];
u3(1.5454924,1.3467702,-1.7948224) q[22];
u3(pi,-1.1071487,0.46364761) q[32];
cx q[32],q[22];
u3(pi/4,0,pi/2) q[22];
u3(pi/4,-pi/2,-pi) q[22];
swap q[22],q[21];
cx q[22],q[23];
u3(pi,-1.1071487,0.46364761) q[22];
u3(1.5454924,1.3467702,-1.7948224) q[23];
cx q[22],q[23];
u3(pi,pi/2,-pi/2) q[22];
u3(pi,0,pi/2) q[22];
u3(3*pi/4,0,-pi/2) q[23];
u3(3*pi/4,pi/2,-pi) q[23];
swap q[23],q[22];
cx q[23],q[33];
u3(pi,-1.1071487,0.46364761) q[23];
cx q[31],q[21];
u3(1.5454924,1.3467702,-1.7948224) q[21];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[21];
u3(3*pi/4,0,-pi/2) q[21];
u3(pi/4,-pi/2,-pi) q[21];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
swap q[21],q[31];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-pi,-pi/2) q[32];
cx q[32],q[22];
u3(1.5454924,1.3467702,-1.7948224) q[22];
u3(pi,-1.1071487,0.46364761) q[32];
cx q[32],q[22];
u3(pi/4,0,pi/2) q[22];
u3(pi/4,-pi/2,-pi) q[22];
cx q[21],q[22];
u3(pi,-1.1071487,0.46364761) q[21];
u3(1.5454924,1.3467702,-1.7948224) q[22];
cx q[21],q[22];
u3(pi,pi/2,-pi/2) q[21];
u3(pi,0,pi/2) q[21];
u3(3*pi/4,0,-pi/2) q[22];
u3(3*pi/4,pi/2,-pi) q[22];
u3(pi,pi/2,-pi/2) q[32];
u3(0,-pi,-pi/2) q[32];
u3(1.5454924,1.3467702,-1.7948224) q[33];
cx q[23],q[33];
u3(2.3428827,-pi/2,pi/2) q[23];
u3(pi/4,0,pi/2) q[33];
u3(3*pi/4,pi/2,-pi) q[33];
cx q[32],q[33];
u3(pi,-1.1071487,0.46364761) q[32];
u3(1.5454924,1.3467702,-1.7948224) q[33];
cx q[32],q[33];
u3(0.79871,pi/2,-pi/2) q[32];
u3(pi/4,0,pi/2) q[33];
u3(3*pi/4,pi/2,-pi) q[33];
swap q[32],q[33];
swap q[22],q[32];
cx q[21],q[22];
u3(pi,-1.1071487,0.46364761) q[21];
u3(1.5454924,1.3467702,-1.7948224) q[22];
cx q[21],q[22];
u3(2.3428827,-pi/2,pi/2) q[21];
u3(pi/4,0,pi/2) q[22];
u3(3*pi/4,pi/2,-pi) q[22];
cx q[39],q[40];
u3(pi,-1.1071487,0.46364761) q[39];
u3(1.5454924,1.3467702,-1.7948224) q[40];
cx q[39],q[40];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
cx q[39],q[31];
u3(1.5454924,1.3467702,-1.7948224) q[31];
u3(pi,-1.1071487,0.46364761) q[39];
cx q[39],q[31];
u3(3*pi/4,0,-pi/2) q[31];
u3(pi/4,-pi/2,-pi) q[31];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
u3(pi/4,0,pi/2) q[40];
u3(3*pi/4,pi/2,-pi) q[40];
swap q[39],q[40];
cx q[38],q[39];
u3(pi,-1.1071487,0.46364761) q[38];
u3(1.5454924,1.3467702,-1.7948224) q[39];
cx q[38],q[39];
u3(pi,pi/2,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
u3(pi/4,0,pi/2) q[39];
u3(pi/4,-pi/2,-pi) q[39];
swap q[38],q[39];
cx q[37],q[38];
u3(pi,-1.1071487,0.46364761) q[37];
u3(1.5454924,1.3467702,-1.7948224) q[38];
cx q[37],q[38];
u3(pi,pi/2,-pi/2) q[37];
u3(0,-pi,-pi/2) q[37];
swap q[37],q[29];
u3(3*pi/4,0,-pi/2) q[38];
u3(3*pi/4,pi/2,-pi) q[38];
cx q[30],q[38];
u3(pi,-1.1071487,0.46364761) q[30];
u3(1.5454924,1.3467702,-1.7948224) q[38];
cx q[30],q[38];
u3(pi,pi/2,-pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi/4,0,pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
cx q[39],q[31];
u3(1.5454924,1.3467702,-1.7948224) q[31];
u3(pi,-1.1071487,0.46364761) q[39];
cx q[39],q[31];
u3(3*pi/4,0,-pi/2) q[31];
u3(pi/4,-pi/2,-pi) q[31];
swap q[31],q[30];
cx q[29],q[30];
u3(pi,-1.1071487,0.46364761) q[29];
u3(1.5454924,1.3467702,-1.7948224) q[30];
cx q[29],q[30];
u3(pi,pi/2,-pi/2) q[29];
u3(0,-pi,-pi/2) q[29];
u3(3*pi/4,0,-pi/2) q[30];
u3(3*pi/4,pi/2,-pi) q[30];
cx q[31],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[30];
u3(pi/4,0,pi/2) q[30];
u3(3*pi/4,pi/2,-pi) q[30];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
cx q[38],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[38];
cx q[38],q[30];
u3(pi/4,0,pi/2) q[30];
u3(0,-pi,-pi/2) q[30];
u3(pi,pi/2,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
cx q[40],q[32];
u3(1.5454924,1.3467702,-1.7948224) q[32];
u3(pi,-1.1071487,0.46364761) q[40];
cx q[40],q[32];
u3(pi/4,0,pi/2) q[32];
u3(3*pi/4,pi/2,-pi) q[32];
u3(pi,pi/2,-pi/2) q[40];
u3(pi,0,pi/2) q[40];
swap q[32],q[40];
cx q[32],q[22];
u3(1.5454924,1.3467702,-1.7948224) q[22];
u3(pi,-1.1071487,0.46364761) q[32];
cx q[32],q[22];
u3(pi/4,0,pi/2) q[22];
u3(3*pi/4,pi/2,-pi) q[22];
u3(2.3428827,-pi/2,pi/2) q[32];
swap q[32],q[22];
cx q[39],q[40];
u3(pi,-1.1071487,0.46364761) q[39];
u3(1.5454924,1.3467702,-1.7948224) q[40];
cx q[39],q[40];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
u3(pi/4,0,pi/2) q[40];
u3(3*pi/4,pi/2,-pi) q[40];
swap q[39],q[40];
swap q[39],q[38];
swap q[30],q[38];
cx q[29],q[30];
u3(pi,-1.1071487,0.46364761) q[29];
u3(1.5454924,1.3467702,-1.7948224) q[30];
cx q[29],q[30];
u3(pi,pi/2,-pi/2) q[29];
u3(pi,0,pi/2) q[29];
u3(pi/4,0,pi/2) q[30];
u3(pi/4,-pi/2,-pi) q[30];
cx q[31],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[30];
u3(3*pi/4,0,-pi/2) q[30];
u3(pi/4,-pi/2,-pi) q[30];
u3(pi,pi/2,-pi/2) q[31];
u3(0,-pi,-pi/2) q[31];
cx q[40],q[32];
u3(1.5454924,1.3467702,-1.7948224) q[32];
u3(pi,-1.1071487,0.46364761) q[40];
cx q[40],q[32];
u3(pi/4,0,pi/2) q[32];
u3(3*pi/4,pi/2,-pi) q[32];
swap q[32],q[31];
swap q[30],q[31];
cx q[29],q[30];
u3(pi,-1.1071487,0.46364761) q[29];
u3(1.5454924,1.3467702,-1.7948224) q[30];
cx q[29],q[30];
u3(2.3428827,-pi/2,pi/2) q[29];
u3(pi/4,0,pi/2) q[30];
u3(3*pi/4,pi/2,-pi) q[30];
cx q[39],q[31];
u3(1.5454924,1.3467702,-1.7948224) q[31];
u3(pi,-1.1071487,0.46364761) q[39];
cx q[39],q[31];
u3(3*pi/4,0,-pi/2) q[31];
u3(3*pi/4,pi/2,-pi) q[31];
swap q[31],q[30];
cx q[32],q[31];
u3(1.5454924,1.3467702,-1.7948224) q[31];
u3(pi,-1.1071487,0.46364761) q[32];
cx q[32],q[31];
u3(pi/4,0,pi/2) q[31];
u3(3*pi/4,pi/2,-pi) q[31];
u3(0.79871,pi/2,-pi/2) q[32];
cx q[38],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[38];
cx q[38],q[30];
u3(pi/4,0,pi/2) q[30];
u3(pi,1.5433012,3.1140976) q[30];
u3(pi,pi/2,-pi/2) q[38];
u3(0,-pi,-pi/2) q[38];
u3(pi,pi/2,-pi/2) q[39];
u3(0,-pi,-pi/2) q[39];
cx q[39],q[31];
u3(1.5454924,1.3467702,-1.7948224) q[31];
u3(pi,-1.1071487,0.46364761) q[39];
cx q[39],q[31];
u3(pi/4,0,pi/2) q[31];
u3(3*pi/4,pi/2,-pi) q[31];
swap q[30],q[31];
cx q[38],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[38];
cx q[38],q[30];
u3(pi/4,0,pi/2) q[30];
u3(3*pi/4,pi/2,-pi) q[30];
cx q[31],q[30];
u3(1.5454924,1.3467702,-1.7948224) q[30];
u3(pi,-1.1071487,0.46364761) q[31];
cx q[31],q[30];
u3(2.0866757,-0.62168558,-2.5386081) q[30];
u3(2.3428827,-pi/2,pi/2) q[31];
u3(0.79871,pi/2,-pi/2) q[38];
u3(0.79871,pi/2,-pi/2) q[39];
u3(0.79871,pi/2,-pi/2) q[40];
barrier q[24],q[34],q[23],q[33],q[21],q[22],q[40],q[29],q[32],q[39],q[38],q[31],q[30];
measure q[24] -> c[0];
measure q[34] -> c[1];
measure q[23] -> c[2];
measure q[33] -> c[3];
measure q[21] -> c[4];
measure q[22] -> c[5];
measure q[40] -> c[6];
measure q[29] -> c[7];
measure q[32] -> c[8];
measure q[39] -> c[9];
measure q[38] -> c[10];
measure q[31] -> c[11];
measure q[30] -> c[12];

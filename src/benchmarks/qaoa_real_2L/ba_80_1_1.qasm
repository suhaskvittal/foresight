OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
creg c[80];
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
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
h q[30];
h q[31];
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
h q[39];
h q[40];
h q[41];
h q[42];
h q[43];
h q[44];
h q[45];
h q[46];
h q[47];
h q[48];
h q[49];
h q[50];
h q[51];
h q[52];
h q[53];
h q[54];
h q[55];
h q[56];
h q[57];
h q[58];
h q[59];
h q[60];
h q[61];
h q[62];
h q[63];
h q[64];
h q[65];
h q[66];
h q[67];
h q[68];
h q[69];
h q[70];
h q[71];
h q[72];
h q[73];
h q[74];
h q[75];
h q[76];
h q[77];
h q[78];
h q[79];
cx q[0],q[1];
rz(-0.237252719383274) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(-0.237252719383274) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(-0.237252719383274) q[5];
cx q[0],q[5];
cx q[0],q[8];
rz(-0.237252719383274) q[8];
cx q[0],q[8];
cx q[0],q[9];
rz(0.237252719383274) q[9];
cx q[0],q[9];
cx q[0],q[23];
rz(0.237252719383274) q[23];
cx q[0],q[23];
cx q[0],q[36];
rz(0.237252719383274) q[36];
cx q[0],q[36];
cx q[0],q[40];
rz(-0.237252719383274) q[40];
cx q[0],q[40];
cx q[0],q[56];
rz(-0.237252719383274) q[56];
cx q[0],q[56];
cx q[0],q[57];
rz(0.237252719383274) q[57];
cx q[0],q[57];
cx q[1],q[2];
rz(-0.237252719383274) q[2];
cx q[1],q[2];
cx q[1],q[15];
rz(-0.237252719383274) q[15];
cx q[1],q[15];
cx q[1],q[54];
rz(0.237252719383274) q[54];
cx q[1],q[54];
cx q[4],q[21];
rz(-0.237252719383274) q[21];
cx q[4],q[21];
cx q[4],q[50];
rz(0.237252719383274) q[50];
cx q[4],q[50];
cx q[4],q[53];
rz(0.237252719383274) q[53];
cx q[4],q[53];
cx q[4],q[58];
rz(-0.237252719383274) q[58];
cx q[4],q[58];
cx q[5],q[7];
rz(-0.237252719383274) q[7];
cx q[5],q[7];
cx q[5],q[12];
rz(0.237252719383274) q[12];
cx q[5],q[12];
cx q[5],q[16];
rz(-0.237252719383274) q[16];
cx q[5],q[16];
cx q[5],q[41];
rz(-0.237252719383274) q[41];
cx q[5],q[41];
cx q[8],q[10];
rz(0.237252719383274) q[10];
cx q[8],q[10];
cx q[8],q[25];
rz(-0.237252719383274) q[25];
cx q[8],q[25];
cx q[8],q[29];
rz(0.237252719383274) q[29];
cx q[8],q[29];
cx q[9],q[13];
rz(0.237252719383274) q[13];
cx q[9],q[13];
cx q[9],q[27];
rz(-0.237252719383274) q[27];
cx q[9],q[27];
cx q[9],q[33];
rz(0.237252719383274) q[33];
cx q[9],q[33];
cx q[9],q[42];
rz(0.237252719383274) q[42];
cx q[9],q[42];
cx q[9],q[48];
rz(0.237252719383274) q[48];
cx q[9],q[48];
cx q[9],q[63];
rz(-0.237252719383274) q[63];
cx q[9],q[63];
cx q[9],q[66];
rz(-0.237252719383274) q[66];
cx q[9],q[66];
cx q[23],q[31];
rz(0.237252719383274) q[31];
cx q[23],q[31];
cx q[23],q[49];
rz(0.237252719383274) q[49];
cx q[23],q[49];
cx q[56],q[74];
rz(0.237252719383274) q[74];
cx q[56],q[74];
cx q[2],q[3];
rz(0.237252719383274) q[3];
cx q[2],q[3];
cx q[2],q[6];
rz(0.237252719383274) q[6];
cx q[2],q[6];
cx q[2],q[11];
rz(-0.237252719383274) q[11];
cx q[2],q[11];
cx q[2],q[18];
rz(0.237252719383274) q[18];
cx q[2],q[18];
cx q[2],q[20];
rz(-0.237252719383274) q[20];
cx q[2],q[20];
cx q[2],q[37];
rz(0.237252719383274) q[37];
cx q[2],q[37];
cx q[2],q[51];
rz(-0.237252719383274) q[51];
cx q[2],q[51];
cx q[2],q[65];
rz(0.237252719383274) q[65];
cx q[2],q[65];
cx q[2],q[73];
rz(-0.237252719383274) q[73];
cx q[2],q[73];
cx q[15],q[17];
rz(-0.237252719383274) q[17];
cx q[15],q[17];
cx q[15],q[55];
rz(0.237252719383274) q[55];
cx q[15],q[55];
cx q[3],q[14];
rz(-0.237252719383274) q[14];
cx q[3],q[14];
cx q[3],q[44];
rz(-0.237252719383274) q[44];
cx q[3],q[44];
cx q[6],q[52];
rz(-0.237252719383274) q[52];
cx q[6],q[52];
cx q[11],q[19];
rz(-0.237252719383274) q[19];
cx q[11],q[19];
cx q[11],q[24];
rz(0.237252719383274) q[24];
cx q[11],q[24];
cx q[11],q[46];
rz(0.237252719383274) q[46];
cx q[11],q[46];
cx q[18],q[43];
rz(0.237252719383274) q[43];
cx q[18],q[43];
cx q[21],q[30];
rz(0.237252719383274) q[30];
cx q[21],q[30];
cx q[21],q[34];
rz(0.237252719383274) q[34];
cx q[21],q[34];
cx q[21],q[35];
rz(-0.237252719383274) q[35];
cx q[21],q[35];
cx q[21],q[62];
rz(-0.237252719383274) q[62];
cx q[21],q[62];
cx q[21],q[78];
rz(-0.237252719383274) q[78];
cx q[21],q[78];
cx q[53],q[59];
rz(-0.237252719383274) q[59];
cx q[53],q[59];
cx q[58],q[72];
rz(0.237252719383274) q[72];
cx q[58],q[72];
cx q[7],q[61];
rz(-0.237252719383274) q[61];
cx q[7],q[61];
cx q[7],q[79];
rz(-0.237252719383274) q[79];
cx q[7],q[79];
cx q[12],q[26];
rz(-0.237252719383274) q[26];
cx q[12],q[26];
cx q[12],q[32];
rz(0.237252719383274) q[32];
cx q[12],q[32];
cx q[52],q[70];
rz(-0.237252719383274) q[70];
cx q[52],q[70];
cx q[13],q[47];
rz(-0.237252719383274) q[47];
cx q[13],q[47];
cx q[13],q[67];
rz(0.237252719383274) q[67];
cx q[13],q[67];
cx q[27],q[28];
rz(-0.237252719383274) q[28];
cx q[27],q[28];
cx q[27],q[45];
rz(-0.237252719383274) q[45];
cx q[27],q[45];
cx q[27],q[76];
rz(0.237252719383274) q[76];
cx q[27],q[76];
cx q[33],q[38];
rz(0.237252719383274) q[38];
cx q[33],q[38];
cx q[33],q[39];
rz(0.237252719383274) q[39];
cx q[33],q[39];
cx q[19],q[22];
rz(-0.237252719383274) q[22];
cx q[19],q[22];
cx q[19],q[64];
rz(0.237252719383274) q[64];
cx q[19],q[64];
cx q[19],q[68];
rz(0.237252719383274) q[68];
cx q[19],q[68];
cx q[46],q[69];
rz(-0.237252719383274) q[69];
cx q[46],q[69];
cx q[17],q[60];
rz(0.237252719383274) q[60];
cx q[17],q[60];
cx q[62],q[71];
rz(-0.237252719383274) q[71];
cx q[62],q[71];
cx q[70],q[75];
rz(-0.237252719383274) q[75];
cx q[70],q[75];
cx q[59],q[77];
rz(-0.237252719383274) q[77];
cx q[59],q[77];
rx(0.169078573986440) q[0];
rx(0.169078573986440) q[1];
rx(0.169078573986440) q[2];
rx(0.169078573986440) q[3];
rx(0.169078573986440) q[4];
rx(0.169078573986440) q[5];
rx(0.169078573986440) q[6];
rx(0.169078573986440) q[7];
rx(0.169078573986440) q[8];
rx(0.169078573986440) q[9];
rx(0.169078573986440) q[10];
rx(0.169078573986440) q[11];
rx(0.169078573986440) q[12];
rx(0.169078573986440) q[13];
rx(0.169078573986440) q[14];
rx(0.169078573986440) q[15];
rx(0.169078573986440) q[16];
rx(0.169078573986440) q[17];
rx(0.169078573986440) q[18];
rx(0.169078573986440) q[19];
rx(0.169078573986440) q[20];
rx(0.169078573986440) q[21];
rx(0.169078573986440) q[22];
rx(0.169078573986440) q[23];
rx(0.169078573986440) q[24];
rx(0.169078573986440) q[25];
rx(0.169078573986440) q[26];
rx(0.169078573986440) q[27];
rx(0.169078573986440) q[28];
rx(0.169078573986440) q[29];
rx(0.169078573986440) q[30];
rx(0.169078573986440) q[31];
rx(0.169078573986440) q[32];
rx(0.169078573986440) q[33];
rx(0.169078573986440) q[34];
rx(0.169078573986440) q[35];
rx(0.169078573986440) q[36];
rx(0.169078573986440) q[37];
rx(0.169078573986440) q[38];
rx(0.169078573986440) q[39];
rx(0.169078573986440) q[40];
rx(0.169078573986440) q[41];
rx(0.169078573986440) q[42];
rx(0.169078573986440) q[43];
rx(0.169078573986440) q[44];
rx(0.169078573986440) q[45];
rx(0.169078573986440) q[46];
rx(0.169078573986440) q[47];
rx(0.169078573986440) q[48];
rx(0.169078573986440) q[49];
rx(0.169078573986440) q[50];
rx(0.169078573986440) q[51];
rx(0.169078573986440) q[52];
rx(0.169078573986440) q[53];
rx(0.169078573986440) q[54];
rx(0.169078573986440) q[55];
rx(0.169078573986440) q[56];
rx(0.169078573986440) q[57];
rx(0.169078573986440) q[58];
rx(0.169078573986440) q[59];
rx(0.169078573986440) q[60];
rx(0.169078573986440) q[61];
rx(0.169078573986440) q[62];
rx(0.169078573986440) q[63];
rx(0.169078573986440) q[64];
rx(0.169078573986440) q[65];
rx(0.169078573986440) q[66];
rx(0.169078573986440) q[67];
rx(0.169078573986440) q[68];
rx(0.169078573986440) q[69];
rx(0.169078573986440) q[70];
rx(0.169078573986440) q[71];
rx(0.169078573986440) q[72];
rx(0.169078573986440) q[73];
rx(0.169078573986440) q[74];
rx(0.169078573986440) q[75];
rx(0.169078573986440) q[76];
rx(0.169078573986440) q[77];
rx(0.169078573986440) q[78];
rx(0.169078573986440) q[79];
cx q[0],q[1];
rz(-0.852275618823128) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(-0.852275618823128) q[4];
cx q[0],q[4];
cx q[0],q[5];
rz(-0.852275618823128) q[5];
cx q[0],q[5];
cx q[0],q[8];
rz(-0.852275618823128) q[8];
cx q[0],q[8];
cx q[0],q[9];
rz(0.852275618823128) q[9];
cx q[0],q[9];
cx q[0],q[23];
rz(0.852275618823128) q[23];
cx q[0],q[23];
cx q[0],q[36];
rz(0.852275618823128) q[36];
cx q[0],q[36];
cx q[0],q[40];
rz(-0.852275618823128) q[40];
cx q[0],q[40];
cx q[0],q[56];
rz(-0.852275618823128) q[56];
cx q[0],q[56];
cx q[0],q[57];
rz(0.852275618823128) q[57];
cx q[0],q[57];
cx q[1],q[2];
rz(-0.852275618823128) q[2];
cx q[1],q[2];
cx q[1],q[15];
rz(-0.852275618823128) q[15];
cx q[1],q[15];
cx q[1],q[54];
rz(0.852275618823128) q[54];
cx q[1],q[54];
cx q[4],q[21];
rz(-0.852275618823128) q[21];
cx q[4],q[21];
cx q[4],q[50];
rz(0.852275618823128) q[50];
cx q[4],q[50];
cx q[4],q[53];
rz(0.852275618823128) q[53];
cx q[4],q[53];
cx q[4],q[58];
rz(-0.852275618823128) q[58];
cx q[4],q[58];
cx q[5],q[7];
rz(-0.852275618823128) q[7];
cx q[5],q[7];
cx q[5],q[12];
rz(0.852275618823128) q[12];
cx q[5],q[12];
cx q[5],q[16];
rz(-0.852275618823128) q[16];
cx q[5],q[16];
cx q[5],q[41];
rz(-0.852275618823128) q[41];
cx q[5],q[41];
cx q[8],q[10];
rz(0.852275618823128) q[10];
cx q[8],q[10];
cx q[8],q[25];
rz(-0.852275618823128) q[25];
cx q[8],q[25];
cx q[8],q[29];
rz(0.852275618823128) q[29];
cx q[8],q[29];
cx q[9],q[13];
rz(0.852275618823128) q[13];
cx q[9],q[13];
cx q[9],q[27];
rz(-0.852275618823128) q[27];
cx q[9],q[27];
cx q[9],q[33];
rz(0.852275618823128) q[33];
cx q[9],q[33];
cx q[9],q[42];
rz(0.852275618823128) q[42];
cx q[9],q[42];
cx q[9],q[48];
rz(0.852275618823128) q[48];
cx q[9],q[48];
cx q[9],q[63];
rz(-0.852275618823128) q[63];
cx q[9],q[63];
cx q[9],q[66];
rz(-0.852275618823128) q[66];
cx q[9],q[66];
cx q[23],q[31];
rz(0.852275618823128) q[31];
cx q[23],q[31];
cx q[23],q[49];
rz(0.852275618823128) q[49];
cx q[23],q[49];
cx q[56],q[74];
rz(0.852275618823128) q[74];
cx q[56],q[74];
cx q[2],q[3];
rz(0.852275618823128) q[3];
cx q[2],q[3];
cx q[2],q[6];
rz(0.852275618823128) q[6];
cx q[2],q[6];
cx q[2],q[11];
rz(-0.852275618823128) q[11];
cx q[2],q[11];
cx q[2],q[18];
rz(0.852275618823128) q[18];
cx q[2],q[18];
cx q[2],q[20];
rz(-0.852275618823128) q[20];
cx q[2],q[20];
cx q[2],q[37];
rz(0.852275618823128) q[37];
cx q[2],q[37];
cx q[2],q[51];
rz(-0.852275618823128) q[51];
cx q[2],q[51];
cx q[2],q[65];
rz(0.852275618823128) q[65];
cx q[2],q[65];
cx q[2],q[73];
rz(-0.852275618823128) q[73];
cx q[2],q[73];
cx q[15],q[17];
rz(-0.852275618823128) q[17];
cx q[15],q[17];
cx q[15],q[55];
rz(0.852275618823128) q[55];
cx q[15],q[55];
cx q[3],q[14];
rz(-0.852275618823128) q[14];
cx q[3],q[14];
cx q[3],q[44];
rz(-0.852275618823128) q[44];
cx q[3],q[44];
cx q[6],q[52];
rz(-0.852275618823128) q[52];
cx q[6],q[52];
cx q[11],q[19];
rz(-0.852275618823128) q[19];
cx q[11],q[19];
cx q[11],q[24];
rz(0.852275618823128) q[24];
cx q[11],q[24];
cx q[11],q[46];
rz(0.852275618823128) q[46];
cx q[11],q[46];
cx q[18],q[43];
rz(0.852275618823128) q[43];
cx q[18],q[43];
cx q[21],q[30];
rz(0.852275618823128) q[30];
cx q[21],q[30];
cx q[21],q[34];
rz(0.852275618823128) q[34];
cx q[21],q[34];
cx q[21],q[35];
rz(-0.852275618823128) q[35];
cx q[21],q[35];
cx q[21],q[62];
rz(-0.852275618823128) q[62];
cx q[21],q[62];
cx q[21],q[78];
rz(-0.852275618823128) q[78];
cx q[21],q[78];
cx q[53],q[59];
rz(-0.852275618823128) q[59];
cx q[53],q[59];
cx q[58],q[72];
rz(0.852275618823128) q[72];
cx q[58],q[72];
cx q[7],q[61];
rz(-0.852275618823128) q[61];
cx q[7],q[61];
cx q[7],q[79];
rz(-0.852275618823128) q[79];
cx q[7],q[79];
cx q[12],q[26];
rz(-0.852275618823128) q[26];
cx q[12],q[26];
cx q[12],q[32];
rz(0.852275618823128) q[32];
cx q[12],q[32];
cx q[52],q[70];
rz(-0.852275618823128) q[70];
cx q[52],q[70];
cx q[13],q[47];
rz(-0.852275618823128) q[47];
cx q[13],q[47];
cx q[13],q[67];
rz(0.852275618823128) q[67];
cx q[13],q[67];
cx q[27],q[28];
rz(-0.852275618823128) q[28];
cx q[27],q[28];
cx q[27],q[45];
rz(-0.852275618823128) q[45];
cx q[27],q[45];
cx q[27],q[76];
rz(0.852275618823128) q[76];
cx q[27],q[76];
cx q[33],q[38];
rz(0.852275618823128) q[38];
cx q[33],q[38];
cx q[33],q[39];
rz(0.852275618823128) q[39];
cx q[33],q[39];
cx q[19],q[22];
rz(-0.852275618823128) q[22];
cx q[19],q[22];
cx q[19],q[64];
rz(0.852275618823128) q[64];
cx q[19],q[64];
cx q[19],q[68];
rz(0.852275618823128) q[68];
cx q[19],q[68];
cx q[46],q[69];
rz(-0.852275618823128) q[69];
cx q[46],q[69];
cx q[17],q[60];
rz(0.852275618823128) q[60];
cx q[17],q[60];
cx q[62],q[71];
rz(-0.852275618823128) q[71];
cx q[62],q[71];
cx q[70],q[75];
rz(-0.852275618823128) q[75];
cx q[70],q[75];
cx q[59],q[77];
rz(-0.852275618823128) q[77];
cx q[59],q[77];
rx(0.00481040507558483) q[0];
rx(0.00481040507558483) q[1];
rx(0.00481040507558483) q[2];
rx(0.00481040507558483) q[3];
rx(0.00481040507558483) q[4];
rx(0.00481040507558483) q[5];
rx(0.00481040507558483) q[6];
rx(0.00481040507558483) q[7];
rx(0.00481040507558483) q[8];
rx(0.00481040507558483) q[9];
rx(0.00481040507558483) q[10];
rx(0.00481040507558483) q[11];
rx(0.00481040507558483) q[12];
rx(0.00481040507558483) q[13];
rx(0.00481040507558483) q[14];
rx(0.00481040507558483) q[15];
rx(0.00481040507558483) q[16];
rx(0.00481040507558483) q[17];
rx(0.00481040507558483) q[18];
rx(0.00481040507558483) q[19];
rx(0.00481040507558483) q[20];
rx(0.00481040507558483) q[21];
rx(0.00481040507558483) q[22];
rx(0.00481040507558483) q[23];
rx(0.00481040507558483) q[24];
rx(0.00481040507558483) q[25];
rx(0.00481040507558483) q[26];
rx(0.00481040507558483) q[27];
rx(0.00481040507558483) q[28];
rx(0.00481040507558483) q[29];
rx(0.00481040507558483) q[30];
rx(0.00481040507558483) q[31];
rx(0.00481040507558483) q[32];
rx(0.00481040507558483) q[33];
rx(0.00481040507558483) q[34];
rx(0.00481040507558483) q[35];
rx(0.00481040507558483) q[36];
rx(0.00481040507558483) q[37];
rx(0.00481040507558483) q[38];
rx(0.00481040507558483) q[39];
rx(0.00481040507558483) q[40];
rx(0.00481040507558483) q[41];
rx(0.00481040507558483) q[42];
rx(0.00481040507558483) q[43];
rx(0.00481040507558483) q[44];
rx(0.00481040507558483) q[45];
rx(0.00481040507558483) q[46];
rx(0.00481040507558483) q[47];
rx(0.00481040507558483) q[48];
rx(0.00481040507558483) q[49];
rx(0.00481040507558483) q[50];
rx(0.00481040507558483) q[51];
rx(0.00481040507558483) q[52];
rx(0.00481040507558483) q[53];
rx(0.00481040507558483) q[54];
rx(0.00481040507558483) q[55];
rx(0.00481040507558483) q[56];
rx(0.00481040507558483) q[57];
rx(0.00481040507558483) q[58];
rx(0.00481040507558483) q[59];
rx(0.00481040507558483) q[60];
rx(0.00481040507558483) q[61];
rx(0.00481040507558483) q[62];
rx(0.00481040507558483) q[63];
rx(0.00481040507558483) q[64];
rx(0.00481040507558483) q[65];
rx(0.00481040507558483) q[66];
rx(0.00481040507558483) q[67];
rx(0.00481040507558483) q[68];
rx(0.00481040507558483) q[69];
rx(0.00481040507558483) q[70];
rx(0.00481040507558483) q[71];
rx(0.00481040507558483) q[72];
rx(0.00481040507558483) q[73];
rx(0.00481040507558483) q[74];
rx(0.00481040507558483) q[75];
rx(0.00481040507558483) q[76];
rx(0.00481040507558483) q[77];
rx(0.00481040507558483) q[78];
rx(0.00481040507558483) q[79];
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
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];
measure q[30] -> c[30];
measure q[31] -> c[31];
measure q[32] -> c[32];
measure q[33] -> c[33];
measure q[34] -> c[34];
measure q[35] -> c[35];
measure q[36] -> c[36];
measure q[37] -> c[37];
measure q[38] -> c[38];
measure q[39] -> c[39];
measure q[40] -> c[40];
measure q[41] -> c[41];
measure q[42] -> c[42];
measure q[43] -> c[43];
measure q[44] -> c[44];
measure q[45] -> c[45];
measure q[46] -> c[46];
measure q[47] -> c[47];
measure q[48] -> c[48];
measure q[49] -> c[49];
measure q[50] -> c[50];
measure q[51] -> c[51];
measure q[52] -> c[52];
measure q[53] -> c[53];
measure q[54] -> c[54];
measure q[55] -> c[55];
measure q[56] -> c[56];
measure q[57] -> c[57];
measure q[58] -> c[58];
measure q[59] -> c[59];
measure q[60] -> c[60];
measure q[61] -> c[61];
measure q[62] -> c[62];
measure q[63] -> c[63];
measure q[64] -> c[64];
measure q[65] -> c[65];
measure q[66] -> c[66];
measure q[67] -> c[67];
measure q[68] -> c[68];
measure q[69] -> c[69];
measure q[70] -> c[70];
measure q[71] -> c[71];
measure q[72] -> c[72];
measure q[73] -> c[73];
measure q[74] -> c[74];
measure q[75] -> c[75];
measure q[76] -> c[76];
measure q[77] -> c[77];
measure q[78] -> c[78];
measure q[79] -> c[79];


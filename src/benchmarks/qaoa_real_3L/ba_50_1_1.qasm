OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
creg c[50];
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
cx q[0],q[1];
rz(-1.77335730344673) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.77335730344673) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(-1.77335730344673) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(-1.77335730344673) q[4];
cx q[0],q[4];
cx q[0],q[6];
rz(-1.77335730344673) q[6];
cx q[0],q[6];
cx q[0],q[11];
rz(-1.77335730344673) q[11];
cx q[0],q[11];
cx q[0],q[12];
rz(-1.77335730344673) q[12];
cx q[0],q[12];
cx q[0],q[13];
rz(1.77335730344673) q[13];
cx q[0],q[13];
cx q[0],q[18];
rz(-1.77335730344673) q[18];
cx q[0],q[18];
cx q[0],q[19];
rz(1.77335730344673) q[19];
cx q[0],q[19];
cx q[0],q[25];
rz(-1.77335730344673) q[25];
cx q[0],q[25];
cx q[0],q[33];
rz(1.77335730344673) q[33];
cx q[0],q[33];
cx q[0],q[39];
rz(-1.77335730344673) q[39];
cx q[0],q[39];
cx q[0],q[44];
rz(-1.77335730344673) q[44];
cx q[0],q[44];
cx q[0],q[49];
rz(-1.77335730344673) q[49];
cx q[0],q[49];
cx q[1],q[7];
rz(-1.77335730344673) q[7];
cx q[1],q[7];
cx q[1],q[38];
rz(-1.77335730344673) q[38];
cx q[1],q[38];
cx q[2],q[5];
rz(-1.77335730344673) q[5];
cx q[2],q[5];
cx q[2],q[21];
rz(1.77335730344673) q[21];
cx q[2],q[21];
cx q[2],q[41];
rz(-1.77335730344673) q[41];
cx q[2],q[41];
cx q[3],q[10];
rz(1.77335730344673) q[10];
cx q[3],q[10];
cx q[3],q[17];
rz(1.77335730344673) q[17];
cx q[3],q[17];
cx q[3],q[27];
rz(-1.77335730344673) q[27];
cx q[3],q[27];
cx q[3],q[29];
rz(1.77335730344673) q[29];
cx q[3],q[29];
cx q[3],q[35];
rz(1.77335730344673) q[35];
cx q[3],q[35];
cx q[3],q[45];
rz(1.77335730344673) q[45];
cx q[3],q[45];
cx q[4],q[9];
rz(-1.77335730344673) q[9];
cx q[4],q[9];
cx q[4],q[24];
rz(1.77335730344673) q[24];
cx q[4],q[24];
cx q[4],q[31];
rz(-1.77335730344673) q[31];
cx q[4],q[31];
cx q[6],q[8];
rz(1.77335730344673) q[8];
cx q[6],q[8];
cx q[6],q[26];
rz(-1.77335730344673) q[26];
cx q[6],q[26];
cx q[12],q[28];
rz(-1.77335730344673) q[28];
cx q[12],q[28];
cx q[13],q[23];
rz(1.77335730344673) q[23];
cx q[13],q[23];
cx q[25],q[40];
rz(1.77335730344673) q[40];
cx q[25],q[40];
cx q[5],q[15];
rz(-1.77335730344673) q[15];
cx q[5],q[15];
cx q[5],q[34];
rz(1.77335730344673) q[34];
cx q[5],q[34];
cx q[5],q[47];
rz(1.77335730344673) q[47];
cx q[5],q[47];
cx q[10],q[14];
rz(1.77335730344673) q[14];
cx q[10],q[14];
cx q[10],q[48];
rz(1.77335730344673) q[48];
cx q[10],q[48];
cx q[29],q[32];
rz(1.77335730344673) q[32];
cx q[29],q[32];
cx q[35],q[37];
rz(1.77335730344673) q[37];
cx q[35],q[37];
cx q[9],q[46];
rz(1.77335730344673) q[46];
cx q[9],q[46];
cx q[24],q[43];
rz(1.77335730344673) q[43];
cx q[24],q[43];
cx q[8],q[16];
rz(-1.77335730344673) q[16];
cx q[8],q[16];
cx q[16],q[20];
rz(-1.77335730344673) q[20];
cx q[16],q[20];
cx q[16],q[30];
rz(-1.77335730344673) q[30];
cx q[16],q[30];
cx q[14],q[22];
rz(-1.77335730344673) q[22];
cx q[14],q[22];
cx q[14],q[42];
rz(1.77335730344673) q[42];
cx q[14],q[42];
cx q[22],q[36];
rz(1.77335730344673) q[36];
cx q[22],q[36];
rx(0.131718061469911) q[0];
rx(0.131718061469911) q[1];
rx(0.131718061469911) q[2];
rx(0.131718061469911) q[3];
rx(0.131718061469911) q[4];
rx(0.131718061469911) q[5];
rx(0.131718061469911) q[6];
rx(0.131718061469911) q[7];
rx(0.131718061469911) q[8];
rx(0.131718061469911) q[9];
rx(0.131718061469911) q[10];
rx(0.131718061469911) q[11];
rx(0.131718061469911) q[12];
rx(0.131718061469911) q[13];
rx(0.131718061469911) q[14];
rx(0.131718061469911) q[15];
rx(0.131718061469911) q[16];
rx(0.131718061469911) q[17];
rx(0.131718061469911) q[18];
rx(0.131718061469911) q[19];
rx(0.131718061469911) q[20];
rx(0.131718061469911) q[21];
rx(0.131718061469911) q[22];
rx(0.131718061469911) q[23];
rx(0.131718061469911) q[24];
rx(0.131718061469911) q[25];
rx(0.131718061469911) q[26];
rx(0.131718061469911) q[27];
rx(0.131718061469911) q[28];
rx(0.131718061469911) q[29];
rx(0.131718061469911) q[30];
rx(0.131718061469911) q[31];
rx(0.131718061469911) q[32];
rx(0.131718061469911) q[33];
rx(0.131718061469911) q[34];
rx(0.131718061469911) q[35];
rx(0.131718061469911) q[36];
rx(0.131718061469911) q[37];
rx(0.131718061469911) q[38];
rx(0.131718061469911) q[39];
rx(0.131718061469911) q[40];
rx(0.131718061469911) q[41];
rx(0.131718061469911) q[42];
rx(0.131718061469911) q[43];
rx(0.131718061469911) q[44];
rx(0.131718061469911) q[45];
rx(0.131718061469911) q[46];
rx(0.131718061469911) q[47];
rx(0.131718061469911) q[48];
rx(0.131718061469911) q[49];
cx q[0],q[1];
rz(-1.10908675708176) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.10908675708176) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(-1.10908675708176) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(-1.10908675708176) q[4];
cx q[0],q[4];
cx q[0],q[6];
rz(-1.10908675708176) q[6];
cx q[0],q[6];
cx q[0],q[11];
rz(-1.10908675708176) q[11];
cx q[0],q[11];
cx q[0],q[12];
rz(-1.10908675708176) q[12];
cx q[0],q[12];
cx q[0],q[13];
rz(1.10908675708176) q[13];
cx q[0],q[13];
cx q[0],q[18];
rz(-1.10908675708176) q[18];
cx q[0],q[18];
cx q[0],q[19];
rz(1.10908675708176) q[19];
cx q[0],q[19];
cx q[0],q[25];
rz(-1.10908675708176) q[25];
cx q[0],q[25];
cx q[0],q[33];
rz(1.10908675708176) q[33];
cx q[0],q[33];
cx q[0],q[39];
rz(-1.10908675708176) q[39];
cx q[0],q[39];
cx q[0],q[44];
rz(-1.10908675708176) q[44];
cx q[0],q[44];
cx q[0],q[49];
rz(-1.10908675708176) q[49];
cx q[0],q[49];
cx q[1],q[7];
rz(-1.10908675708176) q[7];
cx q[1],q[7];
cx q[1],q[38];
rz(-1.10908675708176) q[38];
cx q[1],q[38];
cx q[2],q[5];
rz(-1.10908675708176) q[5];
cx q[2],q[5];
cx q[2],q[21];
rz(1.10908675708176) q[21];
cx q[2],q[21];
cx q[2],q[41];
rz(-1.10908675708176) q[41];
cx q[2],q[41];
cx q[3],q[10];
rz(1.10908675708176) q[10];
cx q[3],q[10];
cx q[3],q[17];
rz(1.10908675708176) q[17];
cx q[3],q[17];
cx q[3],q[27];
rz(-1.10908675708176) q[27];
cx q[3],q[27];
cx q[3],q[29];
rz(1.10908675708176) q[29];
cx q[3],q[29];
cx q[3],q[35];
rz(1.10908675708176) q[35];
cx q[3],q[35];
cx q[3],q[45];
rz(1.10908675708176) q[45];
cx q[3],q[45];
cx q[4],q[9];
rz(-1.10908675708176) q[9];
cx q[4],q[9];
cx q[4],q[24];
rz(1.10908675708176) q[24];
cx q[4],q[24];
cx q[4],q[31];
rz(-1.10908675708176) q[31];
cx q[4],q[31];
cx q[6],q[8];
rz(1.10908675708176) q[8];
cx q[6],q[8];
cx q[6],q[26];
rz(-1.10908675708176) q[26];
cx q[6],q[26];
cx q[12],q[28];
rz(-1.10908675708176) q[28];
cx q[12],q[28];
cx q[13],q[23];
rz(1.10908675708176) q[23];
cx q[13],q[23];
cx q[25],q[40];
rz(1.10908675708176) q[40];
cx q[25],q[40];
cx q[5],q[15];
rz(-1.10908675708176) q[15];
cx q[5],q[15];
cx q[5],q[34];
rz(1.10908675708176) q[34];
cx q[5],q[34];
cx q[5],q[47];
rz(1.10908675708176) q[47];
cx q[5],q[47];
cx q[10],q[14];
rz(1.10908675708176) q[14];
cx q[10],q[14];
cx q[10],q[48];
rz(1.10908675708176) q[48];
cx q[10],q[48];
cx q[29],q[32];
rz(1.10908675708176) q[32];
cx q[29],q[32];
cx q[35],q[37];
rz(1.10908675708176) q[37];
cx q[35],q[37];
cx q[9],q[46];
rz(1.10908675708176) q[46];
cx q[9],q[46];
cx q[24],q[43];
rz(1.10908675708176) q[43];
cx q[24],q[43];
cx q[8],q[16];
rz(-1.10908675708176) q[16];
cx q[8],q[16];
cx q[16],q[20];
rz(-1.10908675708176) q[20];
cx q[16],q[20];
cx q[16],q[30];
rz(-1.10908675708176) q[30];
cx q[16],q[30];
cx q[14],q[22];
rz(-1.10908675708176) q[22];
cx q[14],q[22];
cx q[14],q[42];
rz(1.10908675708176) q[42];
cx q[14],q[42];
cx q[22],q[36];
rz(1.10908675708176) q[36];
cx q[22],q[36];
rx(0.695817207296478) q[0];
rx(0.695817207296478) q[1];
rx(0.695817207296478) q[2];
rx(0.695817207296478) q[3];
rx(0.695817207296478) q[4];
rx(0.695817207296478) q[5];
rx(0.695817207296478) q[6];
rx(0.695817207296478) q[7];
rx(0.695817207296478) q[8];
rx(0.695817207296478) q[9];
rx(0.695817207296478) q[10];
rx(0.695817207296478) q[11];
rx(0.695817207296478) q[12];
rx(0.695817207296478) q[13];
rx(0.695817207296478) q[14];
rx(0.695817207296478) q[15];
rx(0.695817207296478) q[16];
rx(0.695817207296478) q[17];
rx(0.695817207296478) q[18];
rx(0.695817207296478) q[19];
rx(0.695817207296478) q[20];
rx(0.695817207296478) q[21];
rx(0.695817207296478) q[22];
rx(0.695817207296478) q[23];
rx(0.695817207296478) q[24];
rx(0.695817207296478) q[25];
rx(0.695817207296478) q[26];
rx(0.695817207296478) q[27];
rx(0.695817207296478) q[28];
rx(0.695817207296478) q[29];
rx(0.695817207296478) q[30];
rx(0.695817207296478) q[31];
rx(0.695817207296478) q[32];
rx(0.695817207296478) q[33];
rx(0.695817207296478) q[34];
rx(0.695817207296478) q[35];
rx(0.695817207296478) q[36];
rx(0.695817207296478) q[37];
rx(0.695817207296478) q[38];
rx(0.695817207296478) q[39];
rx(0.695817207296478) q[40];
rx(0.695817207296478) q[41];
rx(0.695817207296478) q[42];
rx(0.695817207296478) q[43];
rx(0.695817207296478) q[44];
rx(0.695817207296478) q[45];
rx(0.695817207296478) q[46];
rx(0.695817207296478) q[47];
rx(0.695817207296478) q[48];
rx(0.695817207296478) q[49];
cx q[0],q[1];
rz(-1.79109131207791) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.79109131207791) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(-1.79109131207791) q[3];
cx q[0],q[3];
cx q[0],q[4];
rz(-1.79109131207791) q[4];
cx q[0],q[4];
cx q[0],q[6];
rz(-1.79109131207791) q[6];
cx q[0],q[6];
cx q[0],q[11];
rz(-1.79109131207791) q[11];
cx q[0],q[11];
cx q[0],q[12];
rz(-1.79109131207791) q[12];
cx q[0],q[12];
cx q[0],q[13];
rz(1.79109131207791) q[13];
cx q[0],q[13];
cx q[0],q[18];
rz(-1.79109131207791) q[18];
cx q[0],q[18];
cx q[0],q[19];
rz(1.79109131207791) q[19];
cx q[0],q[19];
cx q[0],q[25];
rz(-1.79109131207791) q[25];
cx q[0],q[25];
cx q[0],q[33];
rz(1.79109131207791) q[33];
cx q[0],q[33];
cx q[0],q[39];
rz(-1.79109131207791) q[39];
cx q[0],q[39];
cx q[0],q[44];
rz(-1.79109131207791) q[44];
cx q[0],q[44];
cx q[0],q[49];
rz(-1.79109131207791) q[49];
cx q[0],q[49];
cx q[1],q[7];
rz(-1.79109131207791) q[7];
cx q[1],q[7];
cx q[1],q[38];
rz(-1.79109131207791) q[38];
cx q[1],q[38];
cx q[2],q[5];
rz(-1.79109131207791) q[5];
cx q[2],q[5];
cx q[2],q[21];
rz(1.79109131207791) q[21];
cx q[2],q[21];
cx q[2],q[41];
rz(-1.79109131207791) q[41];
cx q[2],q[41];
cx q[3],q[10];
rz(1.79109131207791) q[10];
cx q[3],q[10];
cx q[3],q[17];
rz(1.79109131207791) q[17];
cx q[3],q[17];
cx q[3],q[27];
rz(-1.79109131207791) q[27];
cx q[3],q[27];
cx q[3],q[29];
rz(1.79109131207791) q[29];
cx q[3],q[29];
cx q[3],q[35];
rz(1.79109131207791) q[35];
cx q[3],q[35];
cx q[3],q[45];
rz(1.79109131207791) q[45];
cx q[3],q[45];
cx q[4],q[9];
rz(-1.79109131207791) q[9];
cx q[4],q[9];
cx q[4],q[24];
rz(1.79109131207791) q[24];
cx q[4],q[24];
cx q[4],q[31];
rz(-1.79109131207791) q[31];
cx q[4],q[31];
cx q[6],q[8];
rz(1.79109131207791) q[8];
cx q[6],q[8];
cx q[6],q[26];
rz(-1.79109131207791) q[26];
cx q[6],q[26];
cx q[12],q[28];
rz(-1.79109131207791) q[28];
cx q[12],q[28];
cx q[13],q[23];
rz(1.79109131207791) q[23];
cx q[13],q[23];
cx q[25],q[40];
rz(1.79109131207791) q[40];
cx q[25],q[40];
cx q[5],q[15];
rz(-1.79109131207791) q[15];
cx q[5],q[15];
cx q[5],q[34];
rz(1.79109131207791) q[34];
cx q[5],q[34];
cx q[5],q[47];
rz(1.79109131207791) q[47];
cx q[5],q[47];
cx q[10],q[14];
rz(1.79109131207791) q[14];
cx q[10],q[14];
cx q[10],q[48];
rz(1.79109131207791) q[48];
cx q[10],q[48];
cx q[29],q[32];
rz(1.79109131207791) q[32];
cx q[29],q[32];
cx q[35],q[37];
rz(1.79109131207791) q[37];
cx q[35],q[37];
cx q[9],q[46];
rz(1.79109131207791) q[46];
cx q[9],q[46];
cx q[24],q[43];
rz(1.79109131207791) q[43];
cx q[24],q[43];
cx q[8],q[16];
rz(-1.79109131207791) q[16];
cx q[8],q[16];
cx q[16],q[20];
rz(-1.79109131207791) q[20];
cx q[16],q[20];
cx q[16],q[30];
rz(-1.79109131207791) q[30];
cx q[16],q[30];
cx q[14],q[22];
rz(-1.79109131207791) q[22];
cx q[14],q[22];
cx q[14],q[42];
rz(1.79109131207791) q[42];
cx q[14],q[42];
cx q[22],q[36];
rz(1.79109131207791) q[36];
cx q[22],q[36];
rx(0.524388907529432) q[0];
rx(0.524388907529432) q[1];
rx(0.524388907529432) q[2];
rx(0.524388907529432) q[3];
rx(0.524388907529432) q[4];
rx(0.524388907529432) q[5];
rx(0.524388907529432) q[6];
rx(0.524388907529432) q[7];
rx(0.524388907529432) q[8];
rx(0.524388907529432) q[9];
rx(0.524388907529432) q[10];
rx(0.524388907529432) q[11];
rx(0.524388907529432) q[12];
rx(0.524388907529432) q[13];
rx(0.524388907529432) q[14];
rx(0.524388907529432) q[15];
rx(0.524388907529432) q[16];
rx(0.524388907529432) q[17];
rx(0.524388907529432) q[18];
rx(0.524388907529432) q[19];
rx(0.524388907529432) q[20];
rx(0.524388907529432) q[21];
rx(0.524388907529432) q[22];
rx(0.524388907529432) q[23];
rx(0.524388907529432) q[24];
rx(0.524388907529432) q[25];
rx(0.524388907529432) q[26];
rx(0.524388907529432) q[27];
rx(0.524388907529432) q[28];
rx(0.524388907529432) q[29];
rx(0.524388907529432) q[30];
rx(0.524388907529432) q[31];
rx(0.524388907529432) q[32];
rx(0.524388907529432) q[33];
rx(0.524388907529432) q[34];
rx(0.524388907529432) q[35];
rx(0.524388907529432) q[36];
rx(0.524388907529432) q[37];
rx(0.524388907529432) q[38];
rx(0.524388907529432) q[39];
rx(0.524388907529432) q[40];
rx(0.524388907529432) q[41];
rx(0.524388907529432) q[42];
rx(0.524388907529432) q[43];
rx(0.524388907529432) q[44];
rx(0.524388907529432) q[45];
rx(0.524388907529432) q[46];
rx(0.524388907529432) q[47];
rx(0.524388907529432) q[48];
rx(0.524388907529432) q[49];
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

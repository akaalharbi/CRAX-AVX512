#ifndef AVX512_CRAX
#define AVX512_CRAX

#include <immintrin.h>

/* Rotate the bits in each packed 32-bit integer in a to the right by the number of bits specified in imm8, and store the results in dst. */
/* DEFINE RIGHT_ROTATE_DWORDS(src, count_src) { */
/* 	count := count_src % 32 */
/* 	RETURN (src >>count) OR (src << (32 - count)) */
/* } */


#define REG __m512i

#define ROT(V, n) _mm512_ror_epi32 ( (V), (n) )
#define ADD(V, W) _mm512_add_epi32((V), (W))
#define SUB(V, W) _mm512_sub_epi32 ((V), (W))
#define XOR(V, W) _mm512_xor_si512 ( (V), (W) )
#define SET_REG(n) _mm512_set1_epi32((n))





#define  ALZETTE(X, Y, C)				\
  X = ADD(X, ROT(Y, 31)),  Y = XOR(Y, ROT(X, 24)),	\
  X = XOR(X, C),					\
  X = ADD(X, ROT(Y, 17)),  Y = XOR(Y, ROT(X, 17)),	\
  X = XOR(X, C),					\
  X = ADD(X, Y),           Y = XOR(Y, ROT(X, 31)),	\
  X = XOR(X, C),					\
  X = ADD(X, ROT(Y, 24)),  Y = XOR(Y, ROT(X, 16)),	\
  X = XOR(X, C)


#define ALZETTE_INV(X, Y, C )                           \
  X = XOR(X, C),					\
  Y = XOR(Y, ROT(X, 16)), X = SUB(X, ROT(Y, 24)),	\
  X = XOR(X, C),					\
  Y = XOR(Y, ROT(X, 31)), X = SUB(X, Y),                \
  X = XOR(X, C),					\
  Y = XOR(Y, ROT(X, 17)), X = SUB(X, ROT(Y, 17)),	\
  X = XOR(X, C),					\
  Y = XOR(Y, ROT(X, 24)), X = SUB(X, ROT(Y, 31))



REG RCON[5];
void init_crax_rcon() {
  RCON[0] = SET_REG(0xB7E15162);
  RCON[1] = SET_REG(0xBF715880);
  RCON[2] = SET_REG(0x38B4DA56);
  RCON[3] = SET_REG(0x324E7738);
  RCON[4] = SET_REG(0xBB1185EB);
}

REG RXOR_ENC; /* round xor by  i */
REG RXOR_DEC; /* round xor by  i */

int crax_not_inited = 1;

static void crax10_enc_simd(REG* xword, REG* yword, const REG* key)
{
  if (crax_not_inited) {
    init_crax_rcon();
    crax_not_inited = 0;
  }

  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);

  // 10 ROUNDS 
  //  xword[0] = XOR(xword[0], 0); // 0 
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  ALZETTE(xword[0], yword[0], RCON[0]);

  RXOR_ENC = SET_REG(1);
  xword[0] = XOR(xword[0], RXOR_ENC);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  ALZETTE(xword[0], yword[0], RCON[1]);

  RXOR_ENC = _mm512_set1_epi32(2);	      
  xword[0] = XOR(xword[0], RXOR_ENC);//2;
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  ALZETTE(xword[0], yword[0], RCON[2]);

  RXOR_ENC = SET_REG(3);
  xword[0] = XOR(xword[0], RXOR_ENC);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  ALZETTE(xword[0], yword[0], RCON[3]);

  RXOR_ENC = SET_REG(4);
  xword[0] = XOR(xword[0], RXOR_ENC);//
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  ALZETTE(xword[0], yword[0], RCON[4]);

  RXOR_ENC = SET_REG(5);
  xword[0] = XOR(xword[0], RXOR_ENC);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  ALZETTE(xword[0], yword[0], RCON[0]);

  RXOR_ENC = SET_REG(6);
  xword[0] = XOR(xword[0], RXOR_ENC);
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  ALZETTE(xword[0], yword[0], RCON[1]);

  RXOR_ENC = SET_REG(7);
  xword[0] = XOR(xword[0], RXOR_ENC);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  ALZETTE(xword[0], yword[0], RCON[2]);

  RXOR_ENC = SET_REG(8);
  xword[0] = XOR(xword[0], RXOR_ENC);
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  ALZETTE(xword[0], yword[0], RCON[3]);

  RXOR_ENC = SET_REG(9);
  xword[0] = XOR(xword[0], RXOR_ENC);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  ALZETTE(xword[0], yword[0], RCON[4]);  
}


static void crax10_dec_simd(REG* xword, REG* yword, const REG* key)
{
  if (crax_not_inited) {
    init_crax_rcon();
    crax_not_inited = 0;
  }

  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);


  // step = 9
  ALZETTE_INV(xword[0], yword[0], RCON[4]);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  RXOR_ENC = SET_REG(9);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 8
  ALZETTE_INV(xword[0], yword[0], RCON[3]);
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  RXOR_ENC = SET_REG(8);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 7
  ALZETTE_INV(xword[0], yword[0], RCON[2]);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  RXOR_ENC = SET_REG(7);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 6
  ALZETTE_INV(xword[0], yword[0], RCON[1]);
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  RXOR_ENC = SET_REG(6);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 5
  ALZETTE_INV(xword[0], yword[0], RCON[0]);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  RXOR_ENC = SET_REG(5);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 4
  ALZETTE_INV(xword[0], yword[0], RCON[4]);
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  RXOR_ENC = SET_REG(4);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 3
  ALZETTE_INV(xword[0], yword[0], RCON[3]);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  RXOR_ENC = SET_REG(3);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 2
  ALZETTE_INV(xword[0], yword[0], RCON[2]);
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  RXOR_ENC = SET_REG(2);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 1
  ALZETTE_INV(xword[0], yword[0], RCON[1]);
  xword[0] = XOR(xword[0], key[2]);
  yword[0] = XOR(yword[0], key[3]);
  RXOR_ENC = SET_REG(1);
  xword[0] = XOR(xword[0], RXOR_ENC);

  // step = 0
  ALZETTE_INV(xword[0], yword[0], RCON[0]);
  xword[0] = XOR(xword[0], key[0]);
  yword[0] = XOR(yword[0], key[1]);
  /* RXOR_ENC = SET_REG(8); */
  /* xword[0] = XOR(xword[0], RXOR_ENC); */

  

  
}


#endif

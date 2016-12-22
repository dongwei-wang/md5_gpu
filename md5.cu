#include "include.cu"

/* typedef a 32 bit type */
typedef unsigned int UINT4;

/* Data structure for MD5 (Message Digest) computation */
typedef struct __align__(8){
	UINT4 i[2];                 /* number of _bits_ handled mod 2^64 */
	UINT4 buf[4];				/* scratch buffer */
	unsigned char in[64];       /* input buffer */
	unsigned char digest[16];   /* actual digest after MD5Final call */
} MD5_CTX;

__device__ void MD5Init(MD5_CTX *mdContext);
__device__ void MD5Final (MD5_CTX *mdContext);
__device__ static void Transform (UINT4 *buf, UINT4 *in);
__device__ __constant__  unsigned char PADDING[64] = {
	0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
{(a) += F ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}
#define GG(a, b, c, d, x, s, ac) \
{(a) += G ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) \
{(a) += H ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}
#define II(a, b, c, d, x, s, ac) \
{(a) += I ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}

__device__ void MD5Init(MD5_CTX *mdContext){
	mdContext->i[0] = mdContext->i[1] = (UINT4)0;

	/* Load magic initialization constants.
	 */
	mdContext->buf[0] = (UINT4)0x67452301;
	mdContext->buf[1] = (UINT4)0xefcdab89;
	mdContext->buf[2] = (UINT4)0x98badcfe;
	mdContext->buf[3] = (UINT4)0x10325476;
}

__device__ void MD5Final (MD5_CTX *mdContext){
	UINT4 in[16];
	unsigned int i, j;

	/* save number of bits */
	in[14] = mdContext->i[0];
	in[15] = mdContext->i[1];

	/* append length in bits and transform */
	for ( i = 0, j = 0; i < 14; i++, j += 4)
		in[i] = (((UINT4)mdContext->in[j+3]) << 24) |
			(((UINT4)mdContext->in[j+2]) << 16) |
			(((UINT4)mdContext->in[j+1]) << 8) |
			((UINT4)mdContext->in[j]);

	Transform (mdContext->buf, in);

	////////////////////////////////////////////////////////////////////
	/* store buffer in digest */
	for ( i = 0, j = 0; i < 4; i++, j += 4) {
		mdContext->digest[j] = (unsigned char)(mdContext->buf[i] & 0xFF);
		mdContext->digest[j+1] =
			(unsigned char)((mdContext->buf[i] >> 8) & 0xFF);
		mdContext->digest[j+2] =
			(unsigned char)((mdContext->buf[i] >> 16) & 0xFF);
		mdContext->digest[j+3] =
			(unsigned char)((mdContext->buf[i] >> 24) & 0xFF);
	}

	//__syncthreads();

}

// Basic MD5 step. Transform buf based on in.
__device__ static void Transform (UINT4 *buf, UINT4 *in){
	UINT4 a = buf[0], b = buf[1], c = buf[2], d = buf[3];

	// Round 1
#define S11 7
#define S12 12
#define S13 17
#define S14 22
	FF ( a, b, c, d, in[ 0], S11, 3614090360);
	FF ( d, a, b, c, in[ 1], S12, 3905402710);
	FF ( c, d, a, b, in[ 2], S13,  606105819);
	FF ( b, c, d, a, in[ 3], S14, 3250441966);
	FF ( a, b, c, d, in[ 4], S11, 4118548399);
	FF ( d, a, b, c, in[ 5], S12, 1200080426);  //256 Thread OK ????????????
	FF ( c, d, a, b, in[ 6], S13, 2821735955);
	FF ( b, c, d, a, in[ 7], S14, 4249261313);
	FF ( a, b, c, d, in[ 8], S11, 1770035416);
	FF ( d, a, b, c, in[ 9], S12, 2336552879);
	FF ( c, d, a, b, in[10], S13, 4294925233);
	FF ( b, c, d, a, in[11], S14, 2304563134);
	FF ( a, b, c, d, in[12], S11, 1804603682);
	FF ( d, a, b, c, in[13], S12, 4254626195);
	FF ( c, d, a, b, in[14], S13, 2792965006);
	FF ( b, c, d, a, in[15], S14, 1236535329);

	// Round 2
#define S21 5
#define S22 9
#define S23 14
#define S24 20
	GG ( a, b, c, d, in[ 1], S21, 4129170786);
	GG ( d, a, b, c, in[ 6], S22, 3225465664);
	GG ( c, d, a, b, in[11], S23,  643717713);
	GG ( b, c, d, a, in[ 0], S24, 3921069994);
	GG ( a, b, c, d, in[ 5], S21, 3593408605);
	GG ( d, a, b, c, in[10], S22,   38016083);
	GG ( c, d, a, b, in[15], S23, 3634488961);
	GG ( b, c, d, a, in[ 4], S24, 3889429448);
	GG ( a, b, c, d, in[ 9], S21,  568446438);
	GG ( d, a, b, c, in[14], S22, 3275163606);
	GG ( c, d, a, b, in[ 3], S23, 4107603335);
	GG ( b, c, d, a, in[ 8], S24, 1163531501);
	GG ( a, b, c, d, in[13], S21, 2850285829);
	GG ( d, a, b, c, in[ 2], S22, 4243563512);
	GG ( c, d, a, b, in[ 7], S23, 1735328473);
	GG ( b, c, d, a, in[12], S24, 2368359562);

	// Round 3
#define S31 4
#define S32 11
#define S33 16
#define S34 23
	HH ( a, b, c, d, in[ 5], S31, 4294588738);
	HH ( d, a, b, c, in[ 8], S32, 2272392833);
	HH ( c, d, a, b, in[11], S33, 1839030562);
	HH ( b, c, d, a, in[14], S34, 4259657740);
	HH ( a, b, c, d, in[ 1], S31, 2763975236);
	HH ( d, a, b, c, in[ 4], S32, 1272893353);
	HH ( c, d, a, b, in[ 7], S33, 4139469664);
	HH ( b, c, d, a, in[10], S34, 3200236656);
	HH ( a, b, c, d, in[13], S31,  681279174);
	HH ( d, a, b, c, in[ 0], S32, 3936430074);
	HH ( c, d, a, b, in[ 3], S33, 3572445317);
	HH ( b, c, d, a, in[ 6], S34,   76029189);
	HH ( a, b, c, d, in[ 9], S31, 3654602809);
	HH ( d, a, b, c, in[12], S32, 3873151461);
	HH ( c, d, a, b, in[15], S33,  530742520);
	HH ( b, c, d, a, in[ 2], S34, 3299628645);

	// Round 4
#define S41 6
#define S42 10
#define S43 15
#define S44 21
	II ( a, b, c, d, in[ 0], S41, 4096336452);
	II ( d, a, b, c, in[ 7], S42, 1126891415);
	II ( c, d, a, b, in[14], S43, 2878612391);
	II ( b, c, d, a, in[ 5], S44, 4237533241);
	II ( a, b, c, d, in[12], S41, 1700485571);
	II ( d, a, b, c, in[ 3], S42, 2399980690);
	II ( c, d, a, b, in[10], S43, 4293915773);
	II ( b, c, d, a, in[ 1], S44, 2240044497);
	II ( a, b, c, d, in[ 8], S41, 1873313359);
	II ( d, a, b, c, in[15], S42, 4264355552);
	II ( c, d, a, b, in[ 6], S43, 2734768916);
	II ( b, c, d, a, in[13], S44, 1309151649);
	II ( a, b, c, d, in[ 4], S41, 4149444226);
	II ( d, a, b, c, in[11], S42, 3174756917);
	II ( c, d, a, b, in[ 2], S43,  718787259);
	II ( b, c, d, a, in[ 9], S44, 3951481745);

	buf[0] += a;
	buf[1] += b;
	buf[2] += c;
	buf[3] += d;

	//__syncthreads();
}

__device__ __constant__ unsigned char d_table_00_ff[2] = {0x00, 0xFF};
__device__ __constant__ unsigned int d_table_00_fff[2] = {0x00, 0xFFFFFFFF};

__global__ void md5_kernel(int stream_no, unsigned char *d_md5, unsigned int *in, unsigned int *out_md5){
	//CHANNEL_NUM = (THREAD_NUM*THREAD_BLK_NUM) //512*8 =2^12 == 64*64 = 2 ^ 12
	//NUM_PER_THREAD = 131072 //(2^18=1024*128)

	int channel = (blockDim.x*blockIdx.x) + threadIdx.x;
	int matched_id[4];
	matched_id[0]=matched_id[1]=matched_id[2]=matched_id[3]=0;
	int matched_count =0;
	int offset_id=1; //NOTE: start from 1

	unsigned char key_ary[64];
	unsigned char key_base[KEY_LEN];

	unsigned char key_matched[4][MAX_KEY_LEN]; // Max(8,KEY_LEN) = MAX_KEY_LEN

	for (int i=0; i<4; i++){
		for (int j=0; j<MAX_KEY_LEN; j++){
			key_matched[i][j]=0;
		}
	}

	for (int i=0; i<KEY_LEN; i++){
		key_base[i]='A';
		key_ary[i]='A';
	}

	key_base[0] += stream_no;
	key_base[1] += (blockIdx.x <<2) + (threadIdx.x >>5); //2^4 * 2^2
	key_base[2] += ((threadIdx.x & 31) <<1); // 2^5

	key_ary[0] = key_base[0];
	key_ary[1] = key_base[1];
	key_ary[2] = key_base[2];


	for (int i=KEY_LEN; i<64; i++) // Padding.
		key_ary[i]=PADDING[i-KEY_LEN];

	MD5_CTX mdContext;

	unsigned int is_matched=1;

	for (int i=0; i<NUM_PER_THREAD; i++){
		MD5Init (&mdContext);

		mdContext.i[0] = (KEY_LEN<<3);
		for (int j=0; j<64; j++)
			mdContext.in[j] = key_ary[j];
		MD5Final (&mdContext);

		unsigned int all_zero=0;
		for (int j=0; j<16; j++){
			all_zero |= (mdContext.digest[j] ^ d_md5[j]); // is all 0
		}

		is_matched=0; // 0=>0 (!0)=>1
		for (int j=0; j<8; j++){
			is_matched |= ((all_zero>>1) & 1);
			all_zero = all_zero>>1;
		}
		is_matched ^=1;


		matched_id[matched_count] = d_table_00_fff[is_matched] & offset_id;
		for (int j=0; j<KEY_LEN; j++){
			key_matched[matched_count][j] = d_table_00_ff[is_matched] & key_ary[j];
		}
		matched_count = (matched_count + is_matched) & 3; // at most 3.

		key_ary[KEY_LEN-1] = key_base[KEY_LEN-1] + (offset_id &63);
		key_ary[KEY_LEN-2] = key_base[KEY_LEN-2] + ((offset_id >> 6) & 63); // offset_id/64
		key_ary[KEY_LEN-3] = key_base[KEY_LEN-3] + ((offset_id >> 12) &63); // offset_id/64/64

		offset_id++;

		__syncthreads();
	}
	__syncthreads();

	out_md5[(channel<<3)] = (matched_count<<24) + matched_id[0];
	out_md5[(channel<<3)+1] = matched_id[1];
	out_md5[(channel<<3)+2] = matched_id[2];
	out_md5[(channel<<3)+3] = matched_id[3];

	out_md5[(channel<<3)+4] = key_matched[0][0] + (key_matched[0][1]<<8) + (key_matched[0][2]<<16) + (key_matched[0][3]<<24);
	out_md5[(channel<<3)+5] = key_matched[0][4];
	out_md5[(channel<<3)+6] = key_matched[1][0] + (key_matched[1][1]<<8) + (key_matched[1][2]<<16) + (key_matched[1][3]<<24);
	out_md5[(channel<<3)+7] = key_matched[1][4];

	__syncthreads();

	//return;
}

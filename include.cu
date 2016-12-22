#include <stdio.h>
#include <iostream>

#define THREAD_CNT 128 //???? 256, 512 are ERROR
#define BLOCK_CNT 16
#define NUM_PER_THREAD (1024*8)//(2^13)

// STREAM_CNT*RUN_COUNT=64
#define STREAM_CNT 16
#define RUN_COUNT 4
#define CHANNEL_NUM (THREAD_NUM*THREAD_BLK_NUM) //512*8 =2^12
#define KEY_LEN 5
#define MAX_KEY_LEN 8
#define OUTPUT_INT_NUM 8
#define MD5_PASSWD_BYTES_CNT 16

// 'A'=65
//#define MD5_PASSWORD {0xaa,0xd3,0x82,0x53,0xab,0xd5,0xdd,0x13,0x24,0xc3,0x06,0xa2,0x7b,0x77,0xfa,0x0c} // Psswd
#define MD5_PASSWORD {0x31,0x16,0xa8,0x62,0x43,0x25,0x33,0x70,0x44,0xf8,0xe4,0x58,0x37,0xb7,0x13,0x31} // "psswd"
//#define MD5_PASSWORD {0xf6,0xa6,0x26,0x31,0x67,0xc9,0x2d,0xe8,0x64,0x4a,0xc9,0x98,0xb3,0xc4,0xe4,0xd1} // "AAAAA"
//#define MD5_PASSWORD {0x46,0xa0,0xe2,0x40,0xb5,0xe9,0x6a,0x95,0x56,0xfa,0xea,0x8c,0xd9,0xb2,0xcc,0x26} //"AAAAZ"
//#define MD5_PASSWORD {0x87,0xc7,0xd4,0x06,0x8b,0xe0,0x7d,0x39,0x0a,0x1f,0xff,0xd2,0x1b,0xf1,0xe9,0x44} //"BBBBB"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)







#include "include.cu"

__global__ void md5_kernel(int stream_no,unsigned char *d_md5, unsigned int *in,unsigned int * out_md5);

int main(int argc, char *argv[]){
	int nbytes = sizeof(int) * STREAM_CNT * THREAD_CNT * BLOCK_CNT * OUTPUT_INT_NUM;   // number of data bytes
	int bytes_per_stream = sizeof(int) * THREAD_CNT * BLOCK_CNT * OUTPUT_INT_NUM;   // number of bytes in each stream

	// check the compute capability of the device
	int num_devices=0;
	cudaGetDeviceCount(&num_devices);

	if(0 == num_devices){
		printf("your system does not have a CUDA capable device\n");
		return 1;
	}

	// get device information
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, 0);

	if( (1 == device_properties.major) && (device_properties.minor < 1))
		printf("%s does not have compute capability 1.1 or later\n\n", device_properties.name);

	printf("//////////////////// MD5 Crack by GPU ////////////////////////////\n\n");
	printf("Input 16-byte MD5 data array on a 5-char password ranging from 'A~Z,a~z,^-{\\}' : \n\n");

	unsigned char h_md5[] = MD5_PASSWORD;
	for (int i=0; i<MD5_PASSWD_BYTES_CNT; i++)
		printf("%02x", h_md5[i]);
	printf("\n\n");

	//printf("ThreadNum = %d, MD5CountPerThread = %d(K)\n", THREAD_CNT, NUM_PER_THREAD/1024);
	printf("Thread Number		= %d\n", THREAD_CNT);
	printf("MD5CountPerThread	= %d\n", NUM_PER_THREAD);

	//printf("ThreadBlockNum = %d Stream Number = %d\n", BLOCK_CNT, STREAM_CNT);
	printf("Block Number		= %d\n", BLOCK_CNT);
	printf("Stream Number		= %d\n", STREAM_CNT);
	printf("ChannelNum		= %d ( Thread in each block  * Block count )\n", THREAD_CNT*BLOCK_CNT);
	printf("TotalNumber		= %d ( ChannelNum * MD5CountPerThread * StreamNum)\n", NUM_PER_THREAD*THREAD_CNT*BLOCK_CNT*STREAM_CNT);

	// allocate host
	unsigned char *out=0;

	// allocate host memory (pinned is required for achieve asynchronicity)
	cudaMallocHost((void**)&out, nbytes);
	memset(out,0,nbytes);

	// allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t*) malloc(STREAM_CNT * sizeof(cudaStream_t));
	for(int i = 0; i < STREAM_CNT; i++)
		cudaStreamCreate(&(streams[i]));

	// create CUDA event handles
	float elapsed_time=0;   // timing variables
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	// allocate device memory
	unsigned char *d_a = 0, *d_o = 0;             // pointers to data and init value in the device memory

	cudaMalloc((void**)&d_a, nbytes);
	cudaMalloc((void**)&d_o, nbytes);

	cudaMemset(d_a, 0, nbytes);
	cudaMemset(d_o, 0, nbytes);

	unsigned char *d_md5;
	cudaMalloc((void**)&d_md5, MD5_PASSWD_BYTES_CNT);
	cudaMemcpy(d_md5, h_md5, MD5_PASSWD_BYTES_CNT, cudaMemcpyHostToDevice);

	printf("Stream bytes		= %d\n", bytes_per_stream);

	float total_elapsed_time = 0.0f;
	for (int count=0; count<RUN_COUNT; count++){
		cudaMemset(d_a, 0, nbytes);
		cudaMemset(d_o, 0, nbytes);

		cudaEventRecord(start_event, 0);
		// asynchronously launch STREAM_CNT kernels, each operating on its own portion of data
		for(int i = 0; i < STREAM_CNT; i++){
			md5_kernel<<<BLOCK_CNT, THREAD_CNT, 0, streams[i]>>>(count*STREAM_CNT+i,
														d_md5,
														(unsigned int*)(d_a + i * bytes_per_stream),
														(unsigned int*)(d_o + i * bytes_per_stream));
		}

		for(int i = 0; i < STREAM_CNT; i++)
			cudaMemcpyAsync(out + i * bytes_per_stream, d_o + i * bytes_per_stream, bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]);

		cudaEventRecord(stop_event);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		total_elapsed_time += elapsed_time;

		unsigned int * output = (unsigned int *)(out);
		for (int i=0; i<STREAM_CNT*THREAD_CNT*BLOCK_CNT*OUTPUT_INT_NUM; i=i+OUTPUT_INT_NUM){
			if (output[i]!=0){

				printf("ChannelID		= %d\n", i);
				printf("MatchedCount		= %d\n", (output[i] & 0xFF000000)>>24);
				printf("Offset			= %d\n",  (output[i] & 0xFFFFFF)-1);
				printf("Password		= %c%c%c%c%c\n",
						output[i+OUTPUT_INT_NUM/2]&0xFF, (output[i+OUTPUT_INT_NUM/2]&0xFF00)>>8 , (output[i+OUTPUT_INT_NUM/2]&0xFF0000)>>16,
						(output[i+OUTPUT_INT_NUM/2]&0xFF000000)>>24,
						output[i+1+OUTPUT_INT_NUM/2]&0xFF);

				if (((output[i] & 0xFF000000)>>24)>1){
					printf("Offset = %d\n", output[i+1]-1);
					printf("Matched Password = %c%c%c%c%c\n",
							output[i+2+OUTPUT_INT_NUM/2]&0xFF, (output[i+2+OUTPUT_INT_NUM/2]&0xFF00)>>8 , (output[i+2+OUTPUT_INT_NUM/2]&0xFF0000)>>16,
							(output[i+2+OUTPUT_INT_NUM/2]&0xFF000000)>>24,
							output[i+3+OUTPUT_INT_NUM/2]&0xFF );
				}
			}
		}
	}
	printf("GPU Time		= %.4f ms\n", total_elapsed_time);

	// release resources
	for(int i = 0; i < STREAM_CNT; i++)
		cudaStreamDestroy(streams[i]);

	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

	//cudaFreeHost(a);
	cudaFree(d_a);
	cudaFreeHost(out);
	cudaFree(d_o);
	return 0;
}

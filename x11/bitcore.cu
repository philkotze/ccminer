/**
 * Timetravel-10 (bitcore) CUDA implementation
 *  by tpruvot@github - May 2017
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

#define HASH_FUNC_BASE_TIMESTAMP 1492973331 // Bitcore  Genesis
#define HASH_FUNC_COUNT_1 8
#define HASH_FUNC_COUNT_2 8
#define HASH_FUNC_COUNT_3 7
#define HASH_FUNC_COUNT 23
#define HASH_FUNC_VAR_1 3333
#define HASH_FUNC_VAR_2 2100
#define HASH_FUNC_COUNT_PERMUTATIONS_7 5040
#define HASH_FUNC_COUNT_PERMUTATIONS 40320

extern "C" {
#include <sph/sph_blake.h>
#include <sph/sph_bmw.h>
#include <sph/sph_groestl.h>
#include <sph/sph_jh.h>
#include <sph/sph_keccak.h>
#include <sph/sph_skein.h>
#include <sph/sph_luffa.h>
#include <sph/sph_cubehash.h>
#include <sph/sph_shavite.h>
#include <sph/sph_simd.h>
#include <sph/sph_echo.h>
#include <sph/sph_hamsi.h>
#include <sph/sph_fugue.h>
#include <sph/sph_shabal.h>
#include <sph/sph_whirlpool.h>
#include <sph/sph_streebog.h>
#include <sph/sph_haval.h>
#include <sph/sph_sha2.h>
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

static uint32_t* d_hash[MAX_GPUS];

extern void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t* d_hash);

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_nonceVector, uint32_t* d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_nonceVector, uint32_t* d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_nonceVector, uint32_t* d_hash, int order);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_nonceVector, uint32_t* d_hash, int order);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_init(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_hash);

extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_hash, const int outlen);

extern void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_nonceVector, uint32_t* d_hash, int order);

static void swap(int* a, int* b) {
	int c = *a;
	*a = *b;
	*b = c;
}

static void reverse(int* pbegin, int* pend) {
	while ((pbegin != pend) && (pbegin != --pend))
		swap(pbegin++, pend);
}

static void next_permutation(int* pbegin, int* pend) {
	if (pbegin == pend)
		return;

	int* i = pbegin;
	++i;
	if (i == pend)
		return;

	i = pend;
	--i;

	while (1) {
		int* j = i;
		--i;

		if (*i < *j) {
			int* k = pend;

			while (!(*i < *--k))
				/* pass */;

			swap(i, k);
			reverse(j, pend);
			return; // true
		}

		if (i == pbegin) {
			reverse(pbegin, pend);
			return; // false
		}
	}
}

static __thread uint32_t s_ntime = 0;
static uint32_t s_sequence = UINT32_MAX;
static uint8_t s_firstalgo = 0xFF;
static uint8_t hashOrder[HASH_FUNC_COUNT + 1] = { 0 };
#define HASH_FUNC_BASE_TIMESTAMP_1 1492973331 // Bitcore  Genesis
#define HASH_FUNC_COUNT_1 8
#define HASH_FUNC_COUNT_2 8
#define HASH_FUNC_COUNT_3 7
#define HASH_FUNC_COUNT_TOTAL 23
#define HASH_FUNC_VAR_1 3333
#define HASH_FUNC_VAR_2 2100
#define HASH_FUNC_COUNT_PERMUTATIONS_7 5040
#define HASH_FUNC_COUNT_PERMUTATIONS 40320

static __thread int permutation_1[HASH_FUNC_COUNT_1];
static __thread int permutation_2[HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_1];
static __thread int permutation_3[HASH_FUNC_COUNT_3 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_1];
static __thread int algoCount = 0;

static void get_travel_order(uint32_t ntime)
{
	//ntime = 1599098833;
	//char* sptr;

	//applog(LOG_DEBUG, "h2-1");

	//Init1
	for (uint32_t i = 1; i < HASH_FUNC_COUNT_1; i++) {
		permutation_1[i] = i;
	}

	//Init2
	for (uint32_t i = HASH_FUNC_COUNT_1; i < HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_1; i++) {
		permutation_2[i] = i;
	}

	//Init3
	for (uint32_t i = HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2; i < HASH_FUNC_COUNT_3 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_1; i++) {
		permutation_3[i] = i;
	}

	uint32_t steps_1 = (ntime - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
	for (uint32_t i = 0; i < steps_1; i++) {
		next_permutation(permutation_1, permutation_1 + HASH_FUNC_COUNT_1);
	}

	//applog(LOG_DEBUG, "h2-2");

	uint32_t steps_2 = (ntime + HASH_FUNC_VAR_1 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS;
	for (uint32_t i = 0; i < steps_2; i++) {
		next_permutation(permutation_2 + HASH_FUNC_COUNT_1, permutation_2 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2);
	}

	//applog(LOG_DEBUG, "h2-3");

	uint32_t steps_3 = (ntime + HASH_FUNC_VAR_2 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
	for (uint32_t i = 0; i < steps_3; i++) {
		next_permutation(permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2, permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_3);
	}

	//applog(LOG_DEBUG, "h2-4");

	//for (int i = 0; i < 8; i++)
	//	hashOrder[i] = permutation_1[i];
	////	printf("%d\n", permutation_1[i]);

	//for (int i = 8; i < 16; i++)
	//	hashOrder[i] = permutation_2[i];
	////	printf("%d\n", permutation_2[i]);

	//for (int i = 16; i < 23; i++)
	//	hashOrder[i] = permutation_3[i];
	////	printf("%d\n", permutation_3[i]);

	//applog(LOG_DEBUG, "h2-5");
}



// CPU Hash
extern "C" void bitcore_hash(void *output, const void *input)
{
	//applog(LOG_DEBUG, "hi2");
	uint32_t _ALIGN(64) hash[64/4] = { 0 };
	uint32_t s_ntime = UINT32_MAX;

	//applog(LOG_DEBUG, "hi3");

	sph_blake512_context     ctx_blake;
	sph_bmw512_context       ctx_bmw;
	sph_groestl512_context   ctx_groestl;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_skein512_context     ctx_skein;
	sph_luffa512_context     ctx_luffa;
	sph_cubehash512_context  ctx_cubehash;
	sph_shavite512_context   ctx_shavite;
	sph_simd512_context      ctx_simd;
	sph_echo512_context      ctx_echo;
	sph_hamsi512_context     ctx_hamsi;
	sph_fugue512_context     ctx_fugue;
	sph_shabal512_context    ctx_shabal;
	sph_whirlpool_context    ctx_whirlpool;
	sph_sha512_context       ctx_sha512;
	sph_gost512_context      ctx_gost;
	sph_haval256_5_context    ctx_haval;

	//applog(LOG_DEBUG, "hi4");

	if (s_sequence == UINT32_MAX) {
		uint32_t* data = (uint32_t*)input;
		//const uint32_t ntime = (opt_benchmark || !data[17]) ? (uint32_t) time(NULL) : data[17];

		char* sptr;

		//applog(LOG_DEBUG, "hi5");
		const uint32_t ntime = 1599098833;

		//applog(LOG_DEBUG, "hi6");
		for (int i = 0; i < HASH_FUNC_COUNT_1; i++)
			hashOrder[i] = i;

	//	applog(LOG_DEBUG, "hi7");

		uint32_t steps_1 = (ntime - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
		for (uint32_t i = 0; i < steps_1; i++) {
			next_permutation(permutation_1, permutation_1 + HASH_FUNC_COUNT_1);
		}

		//applog(LOG_DEBUG, "hi8");

		uint32_t steps_2 = (ntime + HASH_FUNC_VAR_1 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS;
		for (uint32_t i = 0; i < steps_2; i++) {
			next_permutation(permutation_2 + HASH_FUNC_COUNT_1, permutation_2 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2);
		}

		uint32_t steps_3 = (ntime + HASH_FUNC_VAR_2 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
		for (uint32_t i = 0; i < steps_3; i++) {
			next_permutation(permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2, permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_3);
		}

		//for (int i = 0; i < 8; i++)
		//	sprintf(sptr, "%u", (uint32_t)permutation_1[i]);

		//for (int i = 8; i < 16; i++)
		//	sprintf(sptr, "%u", (uint32_t)permutation_2[i]);

		//for (int i = 16; i < 23; i++)
		//	sprintf(sptr, "%u", (uint32_t)permutation_3[i]);
	}

	int dataLen = 80;
	void* hashA = (void*)input;

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, hashA, dataLen);
	sph_blake512_close(&ctx_blake, hash);

	for (uint32_t i = 1; i < HASH_FUNC_COUNT_1; i++) {
		dataLen = 64;

		switch (permutation_1[i]) {
		case 1:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, hashA, dataLen);
			sph_echo512_close(&ctx_echo, hash);

			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, hash, dataLen);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case 2:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, hashA, dataLen);
			sph_simd512_close(&ctx_simd, hash);

			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, hash, dataLen);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case 3:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, hashA, dataLen);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case 4:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, hashA, dataLen);
			sph_whirlpool_close(&ctx_whirlpool, hash);

			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, hash, dataLen);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case 5:
			sph_gost512_init(&ctx_gost);
			sph_gost512(&ctx_gost, hashA, dataLen);
			sph_gost512_close(&ctx_gost, hash);

			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, hash, dataLen);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case 6:
			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, hashA, dataLen);
			sph_fugue512_close(&ctx_fugue, hash);

			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, hash, dataLen);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case 7:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, hashA, dataLen);
			sph_shavite512_close(&ctx_shavite, hash);

			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, hash, dataLen);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
		}
	}

	for (int i = HASH_FUNC_COUNT_1; i < HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2; i++) {
		switch (permutation_2[i]) {
		case 8:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, hashA, dataLen);
			sph_whirlpool_close(&ctx_whirlpool, hash);

			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, hash, dataLen);
			sph_cubehash512_close(&ctx_cubehash, hash);
			break;
		case 9:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, hashA, dataLen);
			sph_jh512_close(&ctx_jh, hash);

			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, hash, dataLen);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case 10:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, hashA, dataLen);
			sph_blake512_close(&ctx_blake, hash);

			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, hash, dataLen);
			sph_simd512_close(&ctx_simd, hash);
			break;
		case 11:
			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, hashA, dataLen);
			sph_shabal512_close(&ctx_shabal, hash);

			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, hash, dataLen);
			sph_echo512_close(&ctx_echo, hash);
			break;
		case 12:
			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, hashA, dataLen);
			sph_hamsi512_close(&ctx_hamsi, hash);
			break;
		case 13:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, hashA, dataLen);
			sph_bmw512_close(&ctx_bmw, hash);

			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, hash, dataLen);
			sph_fugue512_close(&ctx_fugue, hash);
			break;
		case 14:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, hashA, dataLen);
			sph_keccak512_close(&ctx_keccak, hash);

			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, hash, dataLen);
			sph_shabal512_close(&ctx_shabal, hash);
			break;
		case 15:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, hashA, dataLen);
			sph_luffa512_close(&ctx_luffa, hash);

			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, hash, dataLen);
			sph_whirlpool_close(&ctx_whirlpool, hash);
			break;
		}

	}

	for (int i = HASH_FUNC_COUNT_2; i < HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_3; i++) {
		switch (permutation_3[i]) {
		case 16:
			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512, hashA, dataLen);
			sph_sha512_close(&ctx_sha512, hash);

			sph_haval256_5_init(&ctx_haval);
			sph_haval256_5(&ctx_haval, hash, dataLen);
			sph_haval256_5_close(&ctx_haval, hash);
			break;
		case 17:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, hashA, dataLen);
			sph_skein512_close(&ctx_skein, hash);

			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, hash, dataLen);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case 18:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, hashA, dataLen);
			sph_simd512_close(&ctx_simd, hash);

			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, hash, dataLen);
			sph_hamsi512_close(&ctx_hamsi, hash);
			break;
		case 19:
			sph_gost512_init(&ctx_gost);
			sph_gost512(&ctx_gost, hashA, dataLen);
			sph_gost512_close(&ctx_gost, hash);

			sph_haval256_5_init(&ctx_haval);
			sph_haval256_5(&ctx_haval, hash, dataLen);
			sph_haval256_5_close(&ctx_haval, hash);
			break;
		case 20:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, hashA, dataLen);
			sph_cubehash512_close(&ctx_cubehash, hash);

			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512, hash, dataLen);
			sph_sha512_close(&ctx_sha512, hash);
			break;
		case 21:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, hashA, dataLen);
			sph_echo512_close(&ctx_echo, hash);

			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, hash, dataLen);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case 22:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, hashA, dataLen);
			sph_luffa512_close(&ctx_luffa, hash);

			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, hash, dataLen);
			sph_shabal512_close(&ctx_shabal, hash);
			break;
		}

	}

	memcpy(output, &hash[352], 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "tt-"
#include "cuda_debug.cuh"

void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

void debuglog_hex(void* data, int len)
{
	uint8_t* const bin = (uint8_t*)data;
	char* hex = (char*)calloc(1, len * 2 + 2);
	if (!hex) return;
	for (int i = 0; i < len; i++)
		sprintf(hex + strlen(hex), "%02x", bin[i]);
	strcpy(hex + strlen(hex), "\n");
	printf(hex);
	free(hex);
}

static bool init[MAX_GPUS] = { 0 };
enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	SKEIN,
	JH,
	KECCAK,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HAMSI,
	FUGUE,
	SHABAL,
	WHIRLPOOL,
	SHA512,
	GOST,
	HAVAL,
	MAX_ALGOS_COUNT
};


extern "C" int scanhash_bitcore(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark) pdata[17] = swab32(0x59090909);

	//applog(LOG_DEBUG, "hi");

	if (opt_debug || s_ntime != pdata[17] || s_sequence == UINT32_MAX) {
		uint32_t ntime = swab32(work->data[17]);
		//applog(LOG_DEBUG, "hi1");
		get_travel_order(ntime);
		s_ntime = pdata[17];
		if (opt_debug && !thr_id) {
			applog(LOG_DEBUG, "timetravel10 hash order %s (%08x)", hashOrder, ntime);
		}
	}

	if (opt_benchmark)
		ptarget[7] = 0x5;

	//applog(LOG_DEBUG, "hi1-1");

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		if (x11_simd512_cpu_init(thr_id, throughput) != 0) {
			return 0;
		}
		x11_echo512_cpu_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x17_sha512_cpu_init(thr_id, throughput);
		x17_haval256_cpu_init(thr_id, throughput);

		//applog(LOG_DEBUG, "hi1-2");

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMemset(d_hash[thr_id], 0, (size_t) 64 * throughput), -1);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	//applog(LOG_DEBUG, "hi1-3");

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	cuda_check_cpu_setTarget(ptarget);

	//applog(LOG_DEBUG, "hi1-4");

	// first algo seems locked to blake in bitcore, fine!
	//uint32_t endiandata[20] = { 0 };
	quark_blake512_cpu_setBlock_80(thr_id, endiandata);

	//applog(LOG_DEBUG, "hi1-5");

	do {
		// Hash with CUDA

		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		TRACE("blake80:");

		//applog(LOG_DEBUG, "hi1-6.1");

		//printf("%d\n", d_hash[thr_id]);
		//debuglog_hex(d_hash[thr_id], 64);

		//applog(LOG_DEBUG, "hi1-6.2");

		for (uint32_t i = 1; i < HASH_FUNC_COUNT_1; i++) {
			switch (permutation_1[i]) {
			case 1:
				x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("1a   :");

				quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("1b   :");
				break;
			case 2:
				x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("2a   :");

				quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("2b    :");
				break;
			case 3:
				quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("3a:");
				break;
			case 4:
				x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("4a:");

				quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("4b:");
				break;
			case 5:
				//applog(LOG_DEBUG, "hi1-7");

				streebog_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("5a:");

				//applog(LOG_DEBUG, "hi1-8");

				quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("5b:");

				//applog(LOG_DEBUG, "hi1-9");
				break;
			case 6:
				x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("6a:");

				quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("6b:");
				break;
			case 7:
				x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("7a:");

				x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("7b:");
				break;
			}
		}

		for (int i = HASH_FUNC_COUNT_1; i < HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2; i++) {
			switch (permutation_2[i]) {
			case 8:
				x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("8a:");

				x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("8b:");
				break;
			case 9:
				quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("9a:");

				x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("9b:");
				break;
			case 10:
				quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("10a   :");

				x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("10b   :");
				break;
			case 11:
				x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("11a   :");

				x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("11b   :");
				break;
			case 12:
				x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("12   :");
				break;
			case 13:
				quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("13a   :");

				x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("13b:");
				break;
			case 14:
				quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("14a:");

				x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("14b   :");
				break;
			case 15:
				x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("15a   :");

				x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("15b   :");
				break;
			}
		}

		for (int i = HASH_FUNC_COUNT_2; i < HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_3; i++) {
			switch (permutation_3[i]) {
			case 16:
				x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]);
				TRACE("16a   :");

				x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("16b   :");
				break;
			case 17:
				quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("17a:");

				quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("17b:");
				break;
			case 18:
				x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("18a   :");

				x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("18b   :");
				break;
			case 19:
				streebog_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("19a:");

				x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], 64);
				TRACE("19b:");
				break;
			case 20:
				x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("20a:");

				x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]);
				TRACE("20b   :");
				break;
			case 21:
				x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("21a   :");

				x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("21a:	 :");
				break;
			case 22:
				x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("22a:");

				x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("22b   :");
				break;
			}

		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			be32enc(&endiandata[19], work->nonces[0]);
			bitcore_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				pdata[19] = work->nonces[0];
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					bitcore_hash(vhash, endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
					}
					pdata[19] = max(pdata[19], work->nonces[1]) + 1;
				}
				return work->valid_nonces;
			} else if (vhash[7] > Htarg) {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_bitcore(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}

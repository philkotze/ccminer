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

static uint32_t *d_hash[MAX_GPUS];

inline void swap8(uint8_t *a, uint8_t *b)
{
	uint8_t t = *a;
	*a = *b;
	*b = t;
}


static int nextPerm(uint8_t n[], int count)
{
	int tail, i, j;

	if (count <= 1)
		return 0;

	for (i = count - 1; i>0 && n[i - 1] >= n[i]; i--);
	tail = i;

	if (tail > 0) {
		for (j = count - 1; j>tail && n[j] <= n[tail - 1]; j--);
		swap8(&n[tail - 1], &n[j]);
	}

	for (i = tail, j = count - 1; i<j; i++, j--)
		swap8(&n[i], &n[j]);

	return (tail != 0);
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

static __thread uint8_t permutation_1[HASH_FUNC_COUNT_1];
static __thread uint8_t permutation_2[HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_1];
static __thread uint8_t permutation_3[HASH_FUNC_COUNT_3 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_1];

static void get_travel_order(uint32_t ntime)
{
	ntime = 1599098833;
	char* sptr;

	for (int i = 0; i < HASH_FUNC_COUNT_1; i++)
		hashOrder[i] = i;

	uint32_t steps_1 = (ntime - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
	for (uint32_t i = 0; i < steps_1; i++) {
		nextPerm(permutation_1, (int)permutation_1 + HASH_FUNC_COUNT_1);
	}

	uint32_t steps_2 = (ntime + HASH_FUNC_VAR_1 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS;
	for (uint32_t i = 0; i < steps_2; i++) {
		nextPerm(permutation_2 + HASH_FUNC_COUNT_1, (int)permutation_2 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2);
	}

	uint32_t steps_3 = (ntime + HASH_FUNC_VAR_2 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
	for (uint32_t i = 0; i < steps_3; i++) {
		nextPerm(permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2, (int)permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_3);
	}

	for (int i = 0; i < 8; i++)
		sprintf(sptr, "%u", (uint32_t)permutation_1[i]);

	for (int i = 8; i < 16; i++)
		sprintf(sptr, "%u", (uint32_t)permutation_2[i]);

	for (int i = 16; i < 23; i++)
		sprintf(sptr, "%u", (uint32_t)permutation_3[i]);
}



// CPU Hash
extern "C" void bitcore_hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[64/4] = { 0 };
	uint32_t s_ntime = UINT32_MAX;

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

	if (s_sequence == UINT32_MAX) {
		uint32_t* data = (uint32_t*)input;
		//const uint32_t ntime = (opt_benchmark || !data[17]) ? (uint32_t) time(NULL) : data[17];

		char* sptr;

		const uint32_t ntime = 1599098833;

		for (int i = 0; i < HASH_FUNC_COUNT_1; i++)
			hashOrder[i] = i;

		uint32_t steps_1 = (ntime - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
		for (uint32_t i = 0; i < steps_1; i++) {
			nextPerm(permutation_1, (int)permutation_1 + HASH_FUNC_COUNT_1);
		}

		uint32_t steps_2 = (ntime + HASH_FUNC_VAR_1 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS;
		for (uint32_t i = 0; i < steps_2; i++) {
			nextPerm(permutation_2 + HASH_FUNC_COUNT_1, (int)permutation_2 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2);
		}

		uint32_t steps_3 = (ntime + HASH_FUNC_VAR_2 - HASH_FUNC_BASE_TIMESTAMP_1) % HASH_FUNC_COUNT_PERMUTATIONS_7;
		for (uint32_t i = 0; i < steps_3; i++) {
			nextPerm(permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2, (int)permutation_3 + HASH_FUNC_COUNT_1 + HASH_FUNC_COUNT_2 + HASH_FUNC_COUNT_3);
		}

		for (int i = 0; i < 8; i++)
			sprintf(sptr, "%u", (uint32_t)permutation_1[i]);

		for (int i = 8; i < 16; i++)
			sprintf(sptr, "%u", (uint32_t)permutation_2[i]);

		for (int i = 16; i < 23; i++)
			sprintf(sptr, "%u", (uint32_t)permutation_3[i]);
	}

	void *in = (void*) input;
	int size = 80;

	/*const int hashes = (int) strlen(hashOrder);

	for (int i = 0; i < hashes; i++)
	{
		const char elem = hashOrder[i];
		uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

		if (i > 0) {
			in = (void*) hash;
			size = 64;
		}

		switch (algo) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, in, size);
			sph_cubehash512_close(&ctx_cubehash, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, in, size);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, in, size);
			sph_simd512_close(&ctx_simd, hash);
			break;
#if HASH_FUNC_COUNT > 10
		case ECHO:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, in, size);
			sph_echo512_close(&ctx_echo, hash);
			break;
#endif
		}
	}

	memcpy(output, hash, 32);*/
}

//#define _DEBUG
#define _DEBUG_PREFIX "tt-"
#include "cuda_debug.cuh"

void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

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
#if HASH_FUNC_COUNT > 10
	ECHO,
#endif
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

	if (opt_debug || s_ntime != pdata[17] || s_sequence == UINT32_MAX) {
		uint32_t ntime = swab32(work->data[17]);
		get_travel_order(ntime);
		s_ntime = pdata[17];
		if (opt_debug && !thr_id) {
			applog(LOG_DEBUG, "timetravel10 hash order %s (%08x)", hashOrder, ntime);
		}
	}

	if (opt_benchmark)
		ptarget[7] = 0x5;

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
#if HASH_FUNC_COUNT > 10
		x11_echo512_cpu_init(thr_id, throughput);
#endif
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMemset(d_hash[thr_id], 0, (size_t) 64 * throughput), -1);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	cuda_check_cpu_setTarget(ptarget);

	const int hashes = 40;
	const char first = hashOrder[0];
	const uint8_t algo80 = first >= 'A' ? first - 'A' + 10 : first - '0';
	if (algo80 != s_firstalgo) {
		s_firstalgo = algo80;
	}

	// first algo seems locked to blake in bitcore, fine!
	quark_blake512_cpu_setBlock_80(thr_id, endiandata);

	do {
		// Hash with CUDA

		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		TRACE("blake80:");

		for (int i = 1; i < hashes; i++)
		{
			const char elem = hashOrder[i];
			const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

			switch (algo64) {
			case BLAKE:
				quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("blake  :");
				break;
			case BMW:
				quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("bmw    :");
				break;
			case GROESTL:
				quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("groestl:");
				break;
			case SKEIN:
				quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("skein  :");
				break;
			case JH:
				quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("jh512  :");
				break;
			case KECCAK:
				quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("keccak :");
				break;
			case LUFFA:
				x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("luffa  :");
				break;
			case CUBEHASH:
				x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("cube   :");
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("shavite:");
				break;
			case SIMD:
				x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("simd   :");
				break;
#if HASH_FUNC_COUNT > 10
			case ECHO:
				x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], i);
				TRACE("echo   :");
				break;
#endif
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

#include "sampling.h"
#include "util.h"
#include "algorithm"
#include <math.h>
#include <vector>
#include <queue>
#include <utility>
#include "avx2_target.h"
#include "sampling_avx2.h"
#include "profiling.h"

const int top_k_heap_threshold = 500;

//bool* g_rep_mask = NULL;
//int g_vocab_size = 0;

// Repetition penalty

AVX2_TARGET_OPTIONAL
void apply_rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    const float penalty_max,
    const int sustain,
    const int decay,
    const float alpha_frequency,
    const float alpha_presence,
    const int seq_len,
    float* logits
)
{
    profile_start("apply_rep_penalty_cpu");

    // Map of which logits have already had penalties applied

//    if (vocab_size > g_vocab_size)
//    {
//        if (g_rep_mask) free(g_rep_mask);
//        g_vocab_size = vocab_size;
//        g_rep_mask = (bool*) calloc(1, g_vocab_size * sizeof(bool));
//    }
//    memset(g_rep_mask, 0, g_vocab_size * sizeof(bool));
    bool* g_rep_mask = (bool*) calloc(vocab_size, sizeof(bool));

    // Penalties to apply

    float rep_p = penalty_max;          // Multiplicative penalty, as in HF repetition penalty
    float freq_p = alpha_frequency;     // Additive frequency penalty, as in OAI spec
    float pres_p = alpha_presence;      // Additive presence penalty, as in OAI spec

    // Change in penalties over the "decay" range of the context

    float d_rep_p = 0.0f;
    float d_freq_p = 0.0f;
    float d_pres_p = 0.0f;
    if (decay)
    {
        d_rep_p = (1.0f - rep_p) / (float) decay;
        d_freq_p = (0.0f - freq_p) / (float) decay;
        d_pres_p = (0.0f - pres_p) / (float) decay;
    }

    // "sustain" length, range of the context over which penalties are fixed

    int sust = sustain == -1 ? seq_len : sustain;

    // First token of the penalty range, including decay

    int beg = seq_len - sust - decay;
    if (beg < 0) beg = 0;

    // Iter over context, backwards

    for (int i = seq_len; i > beg;)
    {
        uint64_t t = sequence[--i];
        if (t < vocab_size)
        {

            // If t has not been encountered before, apply rep_p and pres_p

            if (!g_rep_mask[t])
            {
                if (logits[t] > 0.0) logits[t] /= rep_p;  // Multiplicative penalty
                else logits[t] *= rep_p;

                logits[t] -= pres_p;  // Additive penalty

                g_rep_mask[t] = true;  // Only once per logit
            }

            // Apply freq_p penalty for every time a token is encountered, so the total additive penalty is count * freq_p

            logits[t] -= freq_p;
        }

        // If we're in the "decay" range, reduce penalties for every token

        if (--sust < 0)
        {
            rep_p += d_rep_p;
            freq_p += d_freq_p;
            pres_p += d_pres_p;
        }
    }

    free(g_rep_mask);
    profile_stop();
}

AVX2_TARGET_OPTIONAL
int softmax_cpu_nonavx2
(
    const int vocab_size,
    const float temperature,
    const float* logits,
    const bool* logits_filter,
    const float exponent,
    float* output
)
{
    profile_start("softmax_cpu");

    float esum = 0.0f;
    float itemp = 1.0f / temperature;
    float maxl = -1e38;
    int maxi;

    for (int i = 0; i < vocab_size; i++)
    {
        if (logits_filter && !logits_filter[i]) continue;
        if (logits[i] > maxl)
        {
            maxl = logits[i];
            maxi = i;
        }
    }

    for (int i = 0; i < vocab_size; i++)
    {
        if (logits_filter && !logits_filter[i]) continue;
        float l = logits[i] - maxl;
        if (exponent == 2.0f)
            l *= -l;
        else if (exponent != 1.0f)
            l = -powf(fabs(l), exponent);
        float e = expf(l * itemp);
        output[i] = e;
        esum += e;
    }

    float isum = 1.0f / esum;

    for (int i = 0; i < vocab_size; i++)
    {
        if (!logits_filter || logits_filter[i]) output[i] *= isum;
        else output[i] = 0.0f;
    }

    profile_stop();
    return maxi;

//    printf("Softmax:");
//    float summ = 0.0f;
//    for (int i = 0; i < vocab_size; i++)
//    {
//        if (logits_filter[i])
//        {
//            summ += output[i];
//            if (output[i] < 1e-5) continue;
//            printf("%d, %f\n", i, output[i]);
//        }
//    }
//    printf("sum: %f\n\n", summ);
}

int softmax_cpu
(
    const int vocab_size,
    const float temperature,
    const float* logits,
    const bool* logits_filter,
    const float exponent,
    float* output
)
{
    if (is_avx2_supported())
        return softmax_cpu_avx2(vocab_size, temperature, logits, logits_filter, exponent, output);
    else
        return softmax_cpu_nonavx2(vocab_size, temperature, logits, logits_filter, exponent, output);
}

AVX2_TARGET_OPTIONAL
int post_softmax_temperature
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float temperature,
    float min_temp = 0,
    float max_temp = 0.0f,
    float temp_exponent = 1.0f
)
{
    profile_start("post_softmax_temperature");

    if (max_temp > min_temp)
    {
        // Calculate entropy of the softmax probabilities

        float entropy = 0.0f;
        for (int i = 0; i < num_candidates; ++i)
        {
            float prob = temp_probs[i];
            if (prob > 0.0f) entropy -= prob * logf(prob);  // Ensure no log(0)
        }

        // Calculate maximum possible entropy

        float max_entropy = -logf(1.0f / num_candidates);

        // Guard against division by zero

        if (max_entropy == 0.0f) max_entropy = 1.0f;

        // Normalize the entropy

        float normalized_entropy = entropy / max_entropy;

        // Map the normalized entropy to the desired temperature range using the power function

        temperature = min_temp + (max_temp - min_temp) * powf(normalized_entropy, temp_exponent);
    }

//    printf("---- pre\n");
//    for (int i = 0; i < num_candidates; ++i)
//        DBGIF(i, temp_probs[i]);
    
    float psum = 0.0f;
    float itemp = 1.0f / temperature;
    for (int i = 0; i < num_candidates; ++i)
    {
        float p = powf(temp_probs[i], itemp);
        psum += p;
        temp_probs[i] = p;
    }

    float ipsum = 1.0f / psum;
    for (int i = 0; i < num_candidates; ++i)
        temp_probs[i] *= ipsum;

//    printf("---- post\n");
//    DBGF(temperature);
//    printf("----\n");
//    for (int i = 0; i < num_candidates; ++i)
//        DBGIF(i, temp_probs[i]);
//    printf("\n");

    profile_stop();
    return num_candidates;
}

AVX2_TARGET_OPTIONAL
void normalize_cpu
(
    const int num_candidates,
    float* probs
)
{
    profile_start("normalize_cpu");

    float sum = 0.0f;
    #pragma unroll(32)
    for (int i = 0; i < num_candidates; i++) sum += probs[i];
    float isum = 1.0f / sum;
    #pragma unroll(32)
    for (int i = 0; i < num_candidates; i++) probs[i] *= isum;

    profile_stop();
}

template <typename T>
inline void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

inline bool cmp_asc(const float& a, const float& b)
{
    return a > b;
}

inline bool cmp_desc(const float& a, const float& b)
{
    return a < b;
}

template <bool (*cmp_func)(const float&, const float&)>
AVX2_TARGET_OPTIONAL
void quicksort_with_idx
(
    float* arr,
    int* idx,
    int low,
    int high,
    int max_index
)
{
    if (low >= high) return;

    // Bubblesort very short segments

    if (high - low == 1)
    {
        int i0 = low;
        int i1 = low + 1;

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 2)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 3)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;
        int i3 = low + 3;

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i2], arr[i3])) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    if (high - low == 4)
    {
        int i0 = low;
        int i1 = low + 1;
        int i2 = low + 2;
        int i3 = low + 3;
        int i4 = low + 4;

        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i2], arr[i3])) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (cmp_func(arr[i3], arr[i4])) { swap<float>(arr[i3], arr[i4]); swap<int>(idx[i3], idx[i4]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i2], arr[i3])) { swap<float>(arr[i2], arr[i3]); swap<int>(idx[i2], idx[i3]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        if (cmp_func(arr[i1], arr[i2])) { swap<float>(arr[i1], arr[i2]); swap<int>(idx[i1], idx[i2]); }
        if (cmp_func(arr[i0], arr[i1])) { swap<float>(arr[i0], arr[i1]); swap<int>(idx[i0], idx[i1]); }
        return;
    }

    float pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++)
    {
        if (!cmp_func(arr[j], pivot))
        {
            i++;
            swap<float>(arr[i], arr[j]);
            swap<int>(idx[i], idx[j]);
        }
    }

    swap<float>(arr[i + 1], arr[high]);
    swap<int>(idx[i + 1], idx[high]);
    int pos = i + 1;

    if (max_index == 0 || low <= max_index)
        quicksort_with_idx<cmp_func>(arr, idx, low, pos - 1, max_index);
    if (max_index == 0 || pos <= max_index)
        quicksort_with_idx<cmp_func>(arr, idx, pos + 1, high, max_index);
}

// Discard tiny probabilities, improves performance when temperature is very low

AVX2_TARGET_OPTIONAL
int pre_sort_descending
(
    const int num_candidates,
    float* arr,
    int* idx
)
{
    const float eps = 1e-8;
    int i = 0;
    int j = num_candidates - 1;

    while (i <= j)
    {
        if (arr[j] < eps) { j--; continue; }
        if (arr[i] >= eps) { i++; continue; }
        swap<float>(arr[i], arr[j]);
        swap<int>(idx[i], idx[j]);
        i++;
        j--;
    }

    return i;
}

AVX2_TARGET_OPTIONAL
int sort_descending
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int max_index
)
{
    int pre = pre_sort_descending(num_candidates, temp_probs, temp_indices);
    quicksort_with_idx<cmp_desc>(temp_probs, temp_indices, 0, pre - 1, max_index);

//    int m = (max_index == 0 ? num_candidates : max_index);
//    for (int i = 0; i < m; i++) printf("%i - %f \n", temp_indices[i], temp_probs[i] * 10000.0);
//    for (int i = 0; i < m - 1; i++) if (temp_probs[i] < temp_probs[i + 1] - 2e-8) DBGI(i);

    return pre;
}

AVX2_TARGET_OPTIONAL
int top_k_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    int top_k,
    int maxlogit
)
{
    profile_start("top_k_cpu");

    // Special case greedy sampling

    if (top_k == 1)
    {
        if (maxlogit >= 0)
        {
            swap<float>(temp_probs[0], temp_probs[maxlogit]);
            swap<int>(temp_indices[0], temp_indices[maxlogit]);
        }
        else
        {
            int maxidx = -1;
            float max = -1e38;

            for(int i = 0; i < num_candidates; i++)
            {
                if (maxidx == -1 || temp_probs[i] > max)
                {
                    max = temp_probs[i];
                    maxidx = i;
                }
            }

            swap<float>(temp_probs[0], temp_probs[maxidx]);
            swap<int>(temp_indices[0], temp_indices[maxidx]);
        }
    }

    // Use min-heap for lower values of K

    else if (top_k <= top_k_heap_threshold)
    {
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> min_heap;

        for (int i = 0; i < top_k; ++i) min_heap.push({temp_probs[i], temp_indices[i]});

        float t = min_heap.top().first;
        for (int i = top_k; i < num_candidates; i++)
        {
            float p = temp_probs[i];
            if (p > t)
            {
                min_heap.pop();
                min_heap.push({p, temp_indices[i]});
                t = min_heap.top().first;
            }
        }

        int j = top_k;
        for (int i = 0; i < top_k; i++)
        {
            j--;
            temp_probs[j] = min_heap.top().first;
            temp_indices[j] = min_heap.top().second;
            min_heap.pop();
        }
    }

    // For larger values, quicksort is still faster

    else
    {
        sort_descending(num_candidates, temp_probs, temp_indices, top_k);
    }

    profile_stop();
    return top_k;
}

AVX2_TARGET_OPTIONAL
int top_p_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float top_p
)
{
    profile_start("top_p_cpu");

    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> min_heap;

    float min_p = 1e-6;

    float sum = 0.0f;
    for (int i = 0; i < num_candidates; i++)
    {
        if (temp_probs[i] < min_p) continue;
        if (sum > top_p && temp_probs[i] < min_heap.top().first) continue;

        min_heap.push({temp_probs[i], temp_indices[i]});
        sum += temp_probs[i];

        while (sum > top_p && min_heap.size() > 1)
        {
            sum -= min_heap.top().first;
            min_heap.pop();
        }
    }

    int j = min_heap.size();
    int k = j;
    while (j > 0)
    {
        j--;
        temp_probs[j] = min_heap.top().first;
        temp_indices[j] = min_heap.top().second;
        min_heap.pop();
    }

    profile_stop();
    return k;
}

AVX2_TARGET_OPTIONAL
int keep_threshold
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float threshold
)
{
    int i = 0;
    int j = num_candidates - 1;

    while (j >= i)
    {
        while (temp_probs[i] >= threshold && j >= i) i++;
        if (temp_probs[j] >= threshold)
        {
            swap<float>(temp_probs[i], temp_probs[j]);
            swap<int>(temp_indices[i], temp_indices[j]);
            i++;
        }
        j--;
    }
    return i;
}

int top_a_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float top_a
)
{
    profile_start("top_a_cpu");

    // Find top probability
    float top_prob = temp_probs[0];
    for (int i = 1; i < num_candidates; i++)
        if (temp_probs[i] > top_prob) top_prob = temp_probs[i];

    // Calculate the threshold
    float threshold = top_a * top_prob * top_prob;

    // Use the keep_threshold function to keep only probabilities above the threshold
    int n = keep_threshold(num_candidates, temp_probs, temp_indices, threshold);

    profile_stop();
    return n;
}

AVX2_TARGET_OPTIONAL
int min_p_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float min_p
)
{
    profile_start("min_p_cpu");

    float top_prob = temp_probs[0];
    for (int i = 1; i < num_candidates; i++)
        if (temp_probs[i] > top_prob) top_prob = temp_probs[i];

    float threshold = top_prob * min_p;
    int n = keep_threshold(num_candidates, temp_probs, temp_indices, threshold);

    profile_stop();
    return n;
}

AVX2_TARGET_OPTIONAL
int tfs_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float tfs
)
{
    profile_start("tfs_cpu");

    if (num_candidates < 3) return num_candidates;  // Discrete 2nd derivative undefined

    // 2nd derivative of sorted probs

    int nc = sort_descending(num_candidates, temp_probs, temp_indices, num_candidates);

    float* derivative = (float*) calloc(1, nc * sizeof(float));
    float dsum = 0.0f;
    for (int i = 0; i < nc - 2; i++)
    {
        float d = fabs(- temp_probs[i] + 2 * temp_probs[i + 1] - temp_probs[i + 2]);
        dsum += d;
        derivative[i] = d;
    }

    // Keep probs for cumulative sum of normalized 2nd derivative <= threshold

    float dsum_i = 1.0f / dsum;
    int k = 0;
    float cumsum = 0.0f;
    while (k < nc - 2)
    {
        cumsum += derivative[k] * dsum_i;
        if (cumsum > tfs) break;
        k++;
    }

    // Center distribution on the cutoff point

    k++;

    //TIME_STOP;

    free(derivative);
    profile_stop();
    return k;
}

AVX2_TARGET_OPTIONAL
int mirostat_pre_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float mirostat_mu,
    float mirostat_tau,
    float mirostat_eta
)
{
    profile_start("mirostat_pre_cpu");

    // If mu not yet initialized, initialize here

    float mu = mirostat_mu;
    if (mu == 0.0f) mu = mirostat_tau * 2.0f;

    // Discard tokens with surprise greater than mu

    int nc = sort_descending(num_candidates, temp_probs, temp_indices, num_candidates);

    float target_prob = powf(2, -mu);
    int k = 1;
    for (; k < nc; k++)
        if (temp_probs[k] < target_prob) break;

    profile_stop();
    return k;
}

float mirostat_post_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float mirostat_mu,
    float mirostat_tau,
    float mirostat_eta
)
{
    profile_start("mirostat_post_cpu");

    // If mu not yet initializer, initialize here

    float mu = mirostat_mu;
    if (mu == 0.0f) mu = mirostat_tau * 2.0f;

    // Adjust mu based on probability of final choice

    float observed_surprise = -log2(temp_probs[0]);
    mu += mirostat_eta * (mirostat_tau - observed_surprise);

    profile_stop();
    return mu;
}

AVX2_TARGET_OPTIONAL
int typical_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float typical
)
{
    profile_start("typical_cpu");

    const float epsilon = 1e-10;

    int r_candidates = pre_sort_descending(num_candidates, temp_probs, temp_indices);

    float* temp = (float*) calloc(1, r_candidates * sizeof(float));
    int* entropy_dev_order = (int*) calloc(1, r_candidates * sizeof(int));
    int* temp_indices_2 = (int*) calloc(1, r_candidates * sizeof(int));

    float neg_entropy = 0.0f;
    for (int i = 0; i < r_candidates; i++)
    {
        float x = temp_probs[i];
        float y = x + logf(x + epsilon);
        neg_entropy += x * y;
        temp[i] = y;  // temp = log_probs
    }

    for (int i = 0; i < r_candidates; i++)
    {
        temp[i] = fabs(temp[i] - neg_entropy);  // temp = entropy_dev
        entropy_dev_order[i] = i;
    }

    quicksort_with_idx<cmp_asc>(temp, entropy_dev_order, 0, r_candidates - 1, r_candidates);

    memcpy(temp, temp_probs, r_candidates * sizeof(float));  // temp = temp_probs
    memcpy(temp_indices_2, temp_indices, r_candidates * sizeof(int));

    float cumprob = 0.0f;
    int num = 0;

    while (true)
    {
        int j = entropy_dev_order[num];
        float p = temp[j];
        temp_probs[num] = p;
        temp_indices[num] = temp_indices_2[j];

        cumprob += p;
        if (cumprob >= typical) break;
        num++;
        if (num >= r_candidates) break;
    }

    free(temp);
    free(entropy_dev_order);
    free(temp_indices_2);

    if (num == 0) num = 1;

    profile_stop();
    return num;
}

int xtc_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    bool* xtc_mask,
    float xtc_probability,
    float xtc_threshold
)
{
    profile_start("xtc_cpu");

    int index = 0;
    float x_mass = 0.0f;
    int x_tokens = 0;
    float min_prob = 2.0f;
    int min_index = -1;

    // Sum up probs >= xtc_threshold and find the smalles such prob so it can be excluded
    for (int i = 0; i < num_candidates; ++i)
    {
        if (temp_probs[i] >= xtc_threshold && xtc_mask[temp_indices[i]])
        {
            x_tokens++;
            x_mass += temp_probs[i];
            if (temp_probs[i] < min_prob)
            {
                min_prob = temp_probs[i];
                min_index = i;
            }
        }
    }

    // The least probable token over the threshold is never excluded, so do nothing unless at least two tokens
    // cross the threshold
    if (x_tokens <= 1) return num_candidates;

    // Adjust the sum to account for the least probable token being skipped below
    x_mass -= min_prob;

    // Adjust probabilities
    for (int i = 0; i < num_candidates; ++i)
    {
        // If token is being excluded with probability p, its weight in the new distribution is scaled by 1-p
        if (temp_probs[i + 1] >= xtc_threshold && xtc_mask[temp_indices[i]] && i != min_prob)
            temp_probs[i] *= (1.0f - xtc_probability);

        // Scale any tokens we don't remove so so that we end up with a normalized distribution
        else
            temp_probs[i] = temp_probs[i] * (1.0f + xtc_probability * x_mass / (1.0f - x_mass));
    }

    profile_stop();

    // Return unsorted probs since it doesn't affect the multinomial pick. Calling function will sort the probs
    // if necessary
    return num_candidates;
}

AVX2_TARGET_OPTIONAL
int multinomial_cpu
(
    const int num_candidates,
    float* temp_probs,
    int* temp_indices,
    float random
)
{
//    printf("\n-----------------\n");
//    int j = 0;
//    for (int i = 0; i < num_candidates && j < 10; ++i)
//    {
//        if (temp_probs[i] < 1e-6) continue;
//        DBGIF(i, temp_probs[i]);
//        j++;
//    }
//    printf("-----------------\n");

    profile_start("multinomial_cpu");

    int idx = 0;
    float accum = temp_probs[idx];

    while (true)
    {
        if (accum >= random) break;
        if (idx == num_candidates - 1)
        {
            // Roll back in case the sampled probability is exactly zero
            while (idx > 0 && temp_probs[idx] == 0.0f) idx--;
            break;
        }
        idx++;
        accum += temp_probs[idx];
    }

    swap<float>(temp_probs[0], temp_probs[idx]);
    swap<int>(temp_indices[0], temp_indices[idx]);

    profile_stop();
    return 1;
}
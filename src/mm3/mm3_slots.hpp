// src/mm3/mm3_slots.hpp - exact serial prefix parking and idle conversation selection under --parallel.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace mm3 {

struct PictureKey { long long start; uint64_t hash; };
inline bool same_pictures(const std::vector<PictureKey>& a, const std::vector<PictureKey>& b, size_t prefix) {
    const auto included = [&](const std::vector<PictureKey>& x, const std::vector<PictureKey>& y) {
        for (const auto& p : x) if (p.start < (long long) prefix &&
            std::none_of(y.begin(), y.end(), [&](const PictureKey& q) { return p.start == q.start && p.hash == q.hash; }))
            return false;
        return true;
    };
    return included(a, b) && included(b, a);
}
// A parked state is reused only when its entire token prefix and pictures match.
// A partial token match cannot reconstruct DSpark's older sliding-window context.
inline bool exact_prefix(const std::vector<int>& saved, const std::vector<PictureKey>& pictures,
                         const std::vector<int>& prompt, const std::vector<PictureKey>& incoming) {
    return !saved.empty() && saved.size() < prompt.size() &&
           std::equal(saved.begin(), saved.end(), prompt.begin()) && same_pictures(pictures, incoming, saved.size());
}
struct PrefixSlot {
    std::vector<int> tokens;
    std::vector<PictureKey> pictures;
    std::vector<uint8_t> data;
    uint64_t used = 0;
};
class PrefixSlots {
public:
    size_t limit = 4, cap = (size_t) 12 << 30;
    uint64_t clock = 0;
    std::vector<PrefixSlot> slots;
    size_t bytes() const { size_t n = 0; for (const auto& s : slots) n += s.data.size(); return n; }
    int find(const std::vector<int>& prompt, const std::vector<PictureKey>& pictures, size_t better_than = 0) const {
        int best = -1;
        for (int i = 0; i < (int) slots.size(); ++i)
            if (slots[i].tokens.size() > better_than && exact_prefix(slots[i].tokens, slots[i].pictures, prompt, pictures)) {
                best = i; better_than = slots[i].tokens.size();
            }
        return best;
    }
    void drop(size_t i) { slots.erase(slots.begin() + i); }
    bool reserve(size_t n) {
        if (!limit || n > cap) return false;
        while (!slots.empty() && (slots.size() >= limit || bytes() > cap - n)) {
            size_t lru = 0;
            for (size_t i = 1; i < slots.size(); ++i) if (slots[i].used < slots[lru].used) lru = i;
            drop(lru);
        }
        return true;
    }
};

/// One conversation as the choice sees it: busy, the request's shared start with its cache (`from`, the tokens not
/// read again), the tokens in its cache (`size`), and when it last ran (`used`, a clock).
struct SlotView {
    bool busy = false;
    std::size_t from = 0;
    std::size_t size = 0;
    unsigned long long used = 0;
};

/// A shared start this long always counts, whatever it drops (an agent's system prompt, a document asked about again)
constexpr std::size_t kSlotKeepReuse = 512;

/// The conversation a request takes, or -1 (all busy): the most of the request it reuses, then one that drops
/// nothing (empty, or continued), then the least recently used.  A short shared start (under kSlotKeepReuse tokens and
/// under half of what it would drop) counts as none: a short text question shares only the chat template's ~186
/// tokens with a long conversation about a picture, and taking that one's cache read the picture again at its next
/// turn - an empty conversation, else the least recently used, is taken instead.  A continued one counts in full.
inline int pick_slot(const std::vector<SlotView>& v) {
    int best = -1;
    std::size_t b_use = 0;
    bool b_keep = false;
    for (int s = 0; s < (int) v.size(); ++s) {
        const SlotView& x = v[(std::size_t) s];
        if (x.busy) continue;
        const std::size_t drop = x.size > x.from ? x.size - x.from : 0;
        const std::size_t use = drop == 0 || x.from >= kSlotKeepReuse || 2 * x.from >= drop ? x.from : 0;
        const bool keep = drop == 0;
        if (best < 0 || use > b_use ||
            (use == b_use && (keep != b_keep ? keep : x.used < v[(std::size_t) best].used))) {
            best = s;
            b_use = use;
            b_keep = keep;
        }
    }
    return best;
}

}  // namespace mm3

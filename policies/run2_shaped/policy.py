"""the Craftax RL environment policy: PPO network + scripted guard-rails ("survival layer").

Layer 1 (network): the numpy MLP trained by scripts/train.py (weights.npz next to this file).
Layer 2 (rules):  hard facts from the game code, applied on top of the network's action:
  - never step into lava; never sleep unless walled in; never press a craft/place action that
    cannot succeed (saves steps).
  - when drink/food/energy run low, a small scripted routine takes over: walk to water and drink,
    kill and eat a cow or a ripe plant, or dig a 2-tile tunnel into stone, seal it behind us and
    sleep there.
  - remember where the sapling was planted and go back to eat it once it is ripe (600 steps).
  - once we own an iron pickaxe, explore inside mountains for a diamond.
The wrapper keeps a map of everything it has seen (dead-reckoned from its own moves) so it can walk
back to water, the plant, or stone it saw earlier.

Every rule can be switched off via the FLAGS dict below (used for controlled ablations).

AI-use note: designed and drafted with Claude Code from a reading of
craftax_classic/game_logic.py; reviewed, tested and tuned by Yogya Mehrotra.
"""

import heapq
import os

import numpy as np

# ----------------------------------------------------------------------------- constants
(NOOP, LEFT, RIGHT, UP, DOWN, DO, SLEEP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT,
 MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_IRON_PICKAXE, MAKE_WOOD_SWORD, MAKE_STONE_SWORD,
 MAKE_IRON_SWORD) = range(17)
(INVALID, OOB, GRASS, WATER, STONE, TREE, WOOD, PATH, COAL, IRON, DIAMOND, TABLE, FURNACE, SAND,
 LAVA, PLANT, RIPE) = range(17)
H, W, CR, CC = 7, 9, 3, 4  # view size and the player's cell in it
DIRS = {LEFT: (0, -1), RIGHT: (0, 1), UP: (-1, 0), DOWN: (1, 0)}
DIR_OF = {v: k for k, v in DIRS.items()}
OBS_DIR = [LEFT, RIGHT, UP, DOWN]  # obs direction one-hot index -> action
WALK = {GRASS, PATH, SAND}  # player can walk here safely (lava is walkable but deadly)
SOLID = {WATER, STONE, TREE, COAL, IRON, DIAMOND, TABLE, FURNACE, PLANT, RIPE}
ZBLOCK = SOLID | {LAVA, OOB, INVALID}  # zombies/skeletons cannot enter these
ZOMBIE, COW, SKELETON, ARROW = 0, 1, 2, 3
I_WOOD, I_STONE, I_COAL, I_IRON, I_DIAMOND, I_SAPLING = range(6)
I_WPICK, I_SPICK, I_IPICK, I_WSWORD, I_SSWORD, I_ISWORD = range(6, 12)
MEM, OFF = 192, 96  # memory map size; episode starts at (OFF, OFF)
UNKNOWN = -1

FLAGS = dict(
    mask_lava=True, mask_sleep=True, mask_useless=False,
    drink=True, eat=True, shelter=True, plant=True, diamond=True,
    fight_adjacent=True, base=True, topup=False,
)
EARLY_STEPS = int(os.environ.get("CRAFTAX_EARLY", "600"))
EARLY_MODE = os.environ.get("CRAFTAX_EARLYMODE", "none")
# ablations: CRAFTAX_FLAGS="drink=0,shelter=0" (defaults are used for recorded evaluations)
for _kv in os.environ.get("CRAFTAX_FLAGS", "").split(","):
    if "=" in _kv:
        _k, _v = _kv.split("=")
        FLAGS[_k.strip()] = _v.strip() not in ("0", "false", "False")

# ----------------------------------------------------------------------------- network
_w = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz"))
LAYERS = []
_i = 1
while f"w{_i}" in _w:
    LAYERS.append((_w[f"w{_i}"].astype(np.float32), _w[f"b{_i}"].astype(np.float32)))
    _i += 1
WP, BP = _w["wp"].astype(np.float32), _w["bp"].astype(np.float32)


def net_logits(obs):
    h = obs
    for w, b in LAYERS:
        h = np.tanh(h @ w + b)
    return (h @ WP + BP)[:17]


# ----------------------------------------------------------------------------- state
class S:
    pass


s = S()


def reset():
    s.rng = np.random.default_rng(0)
    s.t = 0
    s.pos = (OFF, OFF)
    s.mem = np.full((MEM, MEM), UNKNOWN, np.int8)
    s.last_action = NOOP
    s.last_blocks = None
    s.last_mobs = None
    s.mode = None  # current scripted routine (an object with .step(ctx) -> action or None)
    s.plant = None  # (r, c, t_planted) of our sapling
    s.plant_eaten = False
    s.diamond_done = False
    s.giveup_until = {}  # mode name -> step until which we do not retry it
    s.explore_dir = None
    s.prio = 9
    s.prev_light = 1.0
    s.cow_seen = None
    s.return_to_base = False


reset()


# ----------------------------------------------------------------------------- parsing
class Ctx:
    __slots__ = ("obs", "blocks", "mobs", "inv", "health", "food", "drink", "energy", "facing",
                 "light", "sleeping", "logits")


def parse(obs):
    c = Ctx()
    c.obs = obs
    m = obs[: H * W * 21].reshape(H, W, 21)
    c.blocks = m[:, :, :17].argmax(-1)
    c.mobs = m[:, :, 17:21] > 0.5
    c.inv = np.rint(obs[1323:1335] * 10).astype(int)
    c.health, c.food, c.drink, c.energy = np.rint(obs[1335:1339] * 10).astype(int)
    c.facing = OBS_DIR[int(obs[1339:1343].argmax())]
    c.light = float(obs[1343])
    c.sleeping = obs[1344] > 0.5
    return c


def in_view(vr, vc):
    return 0 <= vr < H and 0 <= vc < W


def world(vr, vc):
    return s.pos[0] + vr - CR, s.pos[1] + vc - CC


def view(r, c):
    return r - s.pos[0] + CR, c - s.pos[1] + CC


def block_at(c, r, col):
    """Block at world (r, col): from the live view if visible, else memory."""
    vr, vc = view(r, col)
    if in_view(vr, vc):
        return int(c.blocks[vr, vc])
    if 0 <= r < MEM and 0 <= col < MEM:
        return int(s.mem[r, col])
    return OOB


def mob_at(c, r, col):
    vr, vc = view(r, col)
    return in_view(vr, vc) and bool(c.mobs[vr, vc].any())


def hostile_adjacent(c):
    for dr, dc in DIRS.values():
        vr, vc = CR + dr, CC + dc
        if c.mobs[vr, vc, ZOMBIE] or c.mobs[vr, vc, SKELETON]:
            return True
    return False


def hostile_near(c, dist=2):
    """A zombie or skeleton within Manhattan distance `dist` of the player."""
    h = c.mobs[:, :, ZOMBIE] | c.mobs[:, :, SKELETON]
    rs, cs = np.nonzero(h)
    return bool(len(rs)) and int((np.abs(rs - CR) + np.abs(cs - CC)).min()) <= dist


def danger(c, r, cc):
    """Tile is next to a zombie/skeleton (planning avoids these)."""
    vr, vc = view(r, cc)
    if not in_view(vr, vc):
        return False
    for dr, dc in DIRS.values():
        nr, nc = vr + dr, vc + dc
        if in_view(nr, nc) and (c.mobs[nr, nc, ZOMBIE] or c.mobs[nr, nc, SKELETON]):
            return True
    return False


def sealed(c, p, q):
    """Two-tile pocket p-q: every neighbour except each other blocks zombies; no mob inside."""
    for a, b in ((p, q), (q, p)):
        if mob_at(c, *a):
            return False
        for dr, dc in DIRS.values():
            n = (a[0] + dr, a[1] + dc)
            if n == b:
                continue
            if block_at(c, *n) not in ZBLOCK or mob_at(c, *n):
                return False
    return True


def enclosed(c):
    for dr, dc in DIRS.values():
        if int(c.blocks[CR + dr, CC + dc]) not in ZBLOCK or c.mobs[CR + dr, CC + dc].any():
            return False
    return True


# ----------------------------------------------------------------------------- odometry + memory
def update_position(c):
    """Dead reckoning: did last step's move succeed? Predict from the old view, verify by overlap."""
    a = s.last_action
    if a in DIRS and s.last_blocks is not None:
        dr, dc = DIRS[a]
        tb = int(s.last_blocks[CR + dr, CC + dc])
        predicted = (tb in WALK or tb == LAVA) and not s.last_mobs[CR + dr, CC + dc].any()
        # verify: compare the new view with the old one under both hypotheses
        def mismatch(sr, sc):
            # hypothesis "we moved by (sr, sc)": new[r, c] must equal old[r + sr, c + sc]
            r0, r1 = max(0, -sr), min(H, H - sr)
            c0, c1 = max(0, -sc), min(W, W - sc)
            new = c.blocks[r0:r1, c0:c1]
            old = s.last_blocks[r0 + sr:r1 + sr, c0 + sc:c1 + sc]
            return int((new != old).sum())

        m_move, m_stay = mismatch(dr, dc), mismatch(0, 0)
        moved = m_move < m_stay if m_move != m_stay else predicted
        if moved:
            s.pos = (s.pos[0] + dr, s.pos[1] + dc)
    r, cc = s.pos
    s.mem[r - CR:r + CR + 1, cc - CC:cc + CC + 1] = c.blocks


# ----------------------------------------------------------------------------- planning
def mineable(c, b):
    if b == STONE or b == COAL:
        return c.inv[I_WPICK] >= 1 if b == COAL else True
    if b == IRON:
        return c.inv[I_SPICK] >= 1
    if b == DIAMOND:
        return c.inv[I_IPICK] >= 1
    return False


def plan(c, goal, allow_mine=False, allow_unknown=False, radius=20, max_nodes=None):
    if max_nodes is None:
        max_nodes = 4000 if radius <= 40 else 12000
    """Dijkstra over the memory map from s.pos. goal(r, c, block) -> bool for a *target* tile:
    the path ends ON the target if it is walkable/mineable, otherwise next to it (we face it).
    Returns list of world tiles after s.pos (the last one may be a solid target), or None."""
    r0, c0 = s.pos
    dist = {s.pos: 0}
    prev = {}
    heap = [(0, s.pos)]
    n = 0
    while heap:
        d, (r, cc) = heapq.heappop(heap)
        if d > dist.get((r, cc), 1e9):
            continue
        n += 1
        if n > max_nodes:
            break
        for dr, dc in DIRS.values():
            nr, nc = r + dr, cc + dc
            if abs(nr - r0) > radius or abs(nc - c0) > radius:
                continue
            b = block_at(c, nr, nc)
            if goal(nr, nc, b):
                path = [(nr, nc)]
                cur = (r, cc)
                while cur != s.pos:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            if mob_at(c, nr, nc):
                continue
            if b in WALK:
                cost = 6 if danger(c, nr, nc) else 1
            elif allow_mine and mineable(c, b):
                cost = 3
            elif allow_unknown and b == UNKNOWN:
                cost = 2
            else:
                continue
            nd = d + cost
            if nd < dist.get((nr, nc), 1e9):
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = (r, cc)
                heapq.heappush(heap, (nd, (nr, nc)))
    return None


def step_along(c, path):
    """Action that advances along path (first tile). None if blocked by something odd."""
    nr, nc = path[0]
    dr, dc = nr - s.pos[0], nc - s.pos[1]
    if (dr, dc) not in DIR_OF:
        return None
    d = DIR_OF[(dr, dc)]
    b = block_at(c, nr, nc)
    if mob_at(c, nr, nc):
        vr, vc = view(nr, nc)
        if c.mobs[vr, vc, COW]:  # a cow in the way: hit it (food!)
            return DO if c.facing == d else d
        return NOOP  # wait for it to move (zombies are handled by the network)
    if b in WALK:
        return d
    if b == LAVA:
        return None
    if mineable(c, b):
        return DO if c.facing == d else d
    return None


# ----------------------------------------------------------------------------- routines
class Routine:
    name = "?"

    cooldown = 60  # steps before this routine may be retried after it gives up

    def __init__(self):
        self.path = None
        self.age = 0

    def done(self, c):
        return True

    def follow(self, c, goal, allow_mine=False, allow_unknown=False, radius=20, replan=8):
        """Walk toward a goal tile; stand next to it facing it. Returns action or None.
        When we are adjacent and facing the goal, returns DO."""
        self.age += 1
        if self.path is None or self.age % replan == 0 or (self.path and self.path[0] == s.pos):
            self.path = plan(c, goal, allow_mine, allow_unknown, radius)
            if self.path is None:
                return None
        if self.path and self.path[0] == s.pos:
            self.path.pop(0)
        if not self.path:
            return None
        if len(self.path) == 1:
            tr, tc = self.path[0]
            b = block_at(c, tr, tc)
            if goal(tr, tc, b):
                d = DIR_OF.get((tr - s.pos[0], tc - s.pos[1]))
                if d is None:
                    self.path = None
                    return None
                if b in WALK and not mob_at(c, tr, tc):
                    return d  # the goal is a floor tile: step onto it
                return DO if c.facing == d else d
        a = step_along(c, self.path)
        if a is None:
            self.path = None
        return a


class Drink(Routine):
    name = "drink"

    def done(self, c):
        return c.drink >= 9

    def step(self, c):
        a = self.follow(c, lambda r, cc, b: b == WATER, allow_unknown=True, radius=60, replan=10)
        if a is not None:
            return a

        def shore_unknown(r, cc, b):
            if b != UNKNOWN:
                return False
            n = sum(block_at(c, r + dr, cc + dc) in (SAND, GRASS) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            return n >= 3

        a = self.follow(c, shore_unknown, allow_unknown=True, radius=30, replan=8)
        return None if a == DO else a


class Eat(Routine):
    name = "eat"

    def done(self, c):
        return c.food >= 9

    def step(self, c):
        # a cow adjacent? hit it
        for d, (dr, dc) in DIRS.items():
            if c.mobs[CR + dr, CC + dc, COW]:
                return DO if c.facing == d else d
        ripe = plan(c, lambda r, cc, b: b == RIPE, radius=10)
        if ripe:
            self.path = None
            return self.follow(c, lambda r, cc, b: b == RIPE, radius=10)
        # walk toward the nearest visible cow
        cows = list(zip(*np.nonzero(c.mobs[:, :, COW])))
        if not cows:
            if s.cow_seen is not None and s.t - s.cow_seen[2] < 300 and \
                    abs(s.cow_seen[0] - s.pos[0]) + abs(s.cow_seen[1] - s.pos[1]) > 2:
                tr, tc, _ = s.cow_seen
                a = self.follow(c, lambda r, cc, b: (r, cc) == (tr, tc), allow_unknown=True,
                                radius=40, replan=10)
                if a is not None and a != DO:
                    return a
                s.cow_seen = None

            def grassy_unknown(r, cc, b):
                if b != UNKNOWN:
                    return False
                n = sum(block_at(c, r + dr, cc + dc) == GRASS for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                return n >= 3

            a = self.follow(c, grassy_unknown, allow_unknown=True, radius=30, replan=8)
            return None if a == DO else a
        vr, vc = min(cows, key=lambda p: abs(p[0] - CR) + abs(p[1] - CC))
        tr, tc = world(vr, vc)
        self.path = None
        return self.follow(c, lambda r, cc, b: (r, cc) == (tr, tc), radius=10, replan=1)


class Shelter(Routine):
    """Dig a 2-deep tunnel into stone, seal the entrance, sleep, then dig out."""
    name = "shelter"

    def __init__(self):
        super().__init__()
        self.site = None  # (E, d) entrance tile and dig direction
        self.tried = set()
        self.last_search = -99
        self.leaving = False
        self.stage = 0
        self.slept = False
        self.wait = 0

    def done(self, c):
        return self.stage >= 9

    def should_stay(self, c):
        """Sealed in: stay while it is useful and we are not starving."""
        if c.drink <= 3 or c.food <= 3:
            return False
        return c.energy < 9 or c.health < 6 or c.light < 0.3

    def site_ok(self, c, e, d):
        """E -> P -> Q in direction d. P and Q are floor or mineable; the four side tiles must
        already block zombies; the far end (beyond Q) may be open if we can spare a stone for it.
        E itself is always sealed from inside with one stone."""
        dr, dc = DIRS[d]
        p = (e[0] + dr, e[1] + dc)
        q = (e[0] + 2 * dr, e[1] + 2 * dc)
        bp, bq = block_at(c, *p), block_at(c, *q)
        okp = bp in WALK or (mineable(c, bp) and bp != DIAMOND)
        okq = bq in WALK or (mineable(c, bq) and bq != DIAMOND)
        if not (okp and okq) or mob_at(c, *p) or mob_at(c, *q):
            return False
        for pr, pc in ((dc, dr), (-dc, -dr)):  # the two sides of the tunnel
            if block_at(c, p[0] + pr, p[1] + pc) not in ZBLOCK:
                return False
            if block_at(c, q[0] + pr, q[1] + pc) not in ZBLOCK:
                return False
        far = block_at(c, q[0] + dr, q[1] + dc)
        needed = 1 + (far not in ZBLOCK)
        if far == UNKNOWN or mob_at(c, q[0] + dr, q[1] + dc):
            return False
        have = c.inv[I_STONE] + (bp == STONE) + (bq == STONE)
        return have >= needed

    def find_site(self, c, radius=40):
        """Nearest known walkable tile E (Manhattan) from which a 2-deep tunnel in some direction
        has solid walls. Candidates are prefiltered with numpy: walkable with a stone 4-neighbour."""
        r0, c0 = s.pos
        lo_r, hi_r = max(1, r0 - radius), min(MEM - 1, r0 + radius + 1)
        lo_c, hi_c = max(1, c0 - radius), min(MEM - 1, c0 + radius + 1)
        m = s.mem[lo_r - 1:hi_r + 1, lo_c - 1:hi_c + 1]
        core = m[1:-1, 1:-1]
        walk = (core == GRASS) | (core == PATH) | (core == SAND)
        st = np.isin(m, list(ZBLOCK - {OOB, INVALID}))
        near = st[:-2, 1:-1] | st[2:, 1:-1] | st[1:-1, :-2] | st[1:-1, 2:]
        rs, cs = np.nonzero(walk & near)
        if len(rs) == 0:
            return None
        rs, cs = rs + lo_r, cs + lo_c
        order = np.argsort(np.abs(rs - r0) + np.abs(cs - c0))
        for i in order[:400]:
            e = (int(rs[i]), int(cs[i]))
            if e in self.tried or mob_at(c, *e):
                continue
            for d in DIRS:
                if self.site_ok(c, e, d):
                    return e, d
        return None

    def step(self, c):
        if c.sleeping:
            self.slept = True
            return NOOP
        if self.stage == 0:  # pick a site and walk to it
            if c.drink <= 2 or c.food <= 2:
                self.cooldown = 0
                return None  # starving: let drink/eat run instead
            if self.site is None:
                if s.t - self.last_search >= 5:
                    self.last_search = s.t
                    self.site = self.find_site(c)
                if self.site is None:  # nothing known: explore toward unknown stone-rich tiles
                    def stony_unknown(r, cc, b):
                        if b != UNKNOWN:
                            return False
                        n = sum(block_at(c, r + dr, cc + dc) == STONE for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                        return n >= 2
                    a = self.follow(c, stony_unknown, allow_unknown=True, radius=30, replan=6)
                    return None if a == DO else a
            e, d = self.site
            if s.pos != e:
                a = self.follow(c, lambda r, cc, b: (r, cc) == e, allow_unknown=True, radius=24)
                if a is None:
                    self.tried.add(e)
                    self.site = None
                return a
            if not self.site_ok(c, e, d):  # the live view disagrees with memory
                self.tried.add(e)
                self.site = None
                return NOOP
            self.stage = 1
        e, d = self.site
        dr, dc = DIRS[d]
        back = DIR_OF[(-dr, -dc)]
        p = (e[0] + dr, e[1] + dc)
        q = (e[0] + 2 * dr, e[1] + 2 * dc)
        # Stateless controller: decide from where we are and what the tiles look like.
        if s.pos == e:
            self.stage = 1
            if block_at(c, *p) in WALK:
                return d
            return DO if c.facing == d else d
        far = (q[0] + dr, q[1] + dc)
        if s.pos == q:
            self.stage = 5
            if block_at(c, *far) not in ZBLOCK:  # open far end: seal it first
                if c.facing == d:
                    return PLACE_STONE if c.inv[I_STONE] >= 1 else None
                return back  # go to P and come back facing d
            return back
        if s.pos == p:
            if block_at(c, *q) not in WALK:
                self.stage = 3
                return DO if c.facing == d else d
            if block_at(c, *far) not in ZBLOCK:
                self.stage = 4
                return d  # step into Q facing the open far end
            if block_at(c, *e) in ZBLOCK:  # sealed in
                if not sealed(c, p, q):
                    self.stage, self.site = 0, None
                    return NOOP
                if self.leaving or not self.should_stay(c):
                    self.leaving = True
                    self.stage = 8
                    return DO if c.facing == back else back  # dig out through E
                self.stage = 7
                return SLEEP if c.energy < 9 else NOOP
            if self.leaving:  # E is open again: done
                self.stage = 9
                return NOOP
            if c.facing == back:
                self.stage = 6
                if c.inv[I_STONE] < 1:
                    self.stage, self.site = 0, None
                    return NOOP
                return PLACE_STONE
            self.stage = 4
            return d  # go to Q and come back facing E
        # somewhere else (a fight moved us): walk back to E
        self.stage = 0
        return self.follow(c, lambda r, cc, b: (r, cc) == e, allow_unknown=True, radius=24)


class Base(Routine):
    """The network has no memory: after a scripted detour, bring it back next to its table."""
    name = "base"

    def done(self, c):
        return (c.blocks == TABLE).any()

    def step(self, c):
        return self.follow(c, lambda r, cc, b: b == TABLE, allow_unknown=True, radius=40, replan=10)


class EatPlant(Routine):
    name = "plant"

    def done(self, c):
        return s.plant_eaten or s.plant is None

    def step(self, c):
        pr, pc, _ = s.plant
        b = block_at(c, pr, pc)
        if b not in (PLANT, RIPE, UNKNOWN):
            s.plant = None  # it is gone
            return None
        a = self.follow(c, lambda r, cc, bb: (r, cc) == (pr, pc), allow_mine=True,
                        allow_unknown=True, radius=90, replan=12)
        if a == DO and b != RIPE:
            return NOOP  # wait next to it
        return a


class Diamond(Routine):
    name = "diamond"

    def __init__(self):
        super().__init__()
        self.target = None

    def done(self, c):
        return s.diamond_done

    def step(self, c):
        if c.inv[I_IPICK] < 1:
            return None
        known = plan(c, lambda r, cc, b: b == DIAMOND, allow_mine=True, radius=30)
        if known:
            self.path = None
            return self.follow(c, lambda r, cc, b: b == DIAMOND, allow_mine=True, radius=30)

        def frontier(r, cc, b):
            if b != UNKNOWN:
                return False
            stone = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = block_at(c, r + dr, cc + dc)
                    stone += nb in (STONE, COAL, IRON, PATH)
            return stone >= 4

        a = self.follow(c, frontier, allow_mine=True, radius=24, replan=10)
        if a is None:
            a = self.follow(c, lambda r, cc, b: b == UNKNOWN, allow_mine=True, radius=24, replan=10)
        return a


# ----------------------------------------------------------------------------- masks
def masked_sample(c):
    logits = net_logits(c.obs).astype(np.float64)
    ok = np.ones(17, bool)
    f, (fr, fc) = c.facing, DIRS[c.facing]
    front = int(c.blocks[CR + fr, CC + fc])
    near_table = any(int(c.blocks[CR + dr, CC + dc]) == TABLE for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    near_furn = any(int(c.blocks[CR + dr, CC + dc]) == FURNACE for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    if FLAGS["mask_lava"]:
        for d, (dr, dc) in DIRS.items():
            if int(c.blocks[CR + dr, CC + dc]) == LAVA:
                ok[d] = False
    tree_done = c.inv[I_IPICK] >= 1 and (c.inv[I_ISWORD] >= 1 or s.t >= EARLY_STEPS)
    if FLAGS["mask_sleep"] and tree_done and not enclosed(c):
        hostile_seen = c.mobs[:, :, ZOMBIE].any() or c.mobs[:, :, SKELETON].any()
        if not (c.light > 0.6 and not hostile_seen and c.energy < 9):
            ok[SLEEP] = False
    if FLAGS["mask_useless"]:
        inv = c.inv
        placeable = front in (GRASS, SAND, PATH, WATER, LAVA)
        ok[PLACE_STONE] &= inv[I_STONE] >= 1 and placeable
        ok[PLACE_TABLE] &= inv[I_WOOD] >= 2 and front not in SOLID and front != OOB
        ok[PLACE_FURNACE] &= inv[I_STONE] >= 1 and near_table and front not in SOLID and front != OOB
        ok[PLACE_PLANT] &= inv[I_SAPLING] >= 1 and front == GRASS
        ok[MAKE_WOOD_PICKAXE] &= inv[I_WPICK] < 1 and inv[I_WOOD] >= 1 and near_table
        ok[MAKE_STONE_PICKAXE] &= inv[I_SPICK] < 1 and inv[I_WOOD] >= 1 and inv[I_STONE] >= 1 and near_table
        ok[MAKE_IRON_PICKAXE] &= (inv[I_IPICK] < 1 and inv[I_WOOD] >= 1 and inv[I_STONE] >= 1
                                  and inv[I_COAL] >= 1 and inv[I_IRON] >= 1 and near_table and near_furn)
        ok[MAKE_WOOD_SWORD] &= inv[I_WSWORD] < 1 and inv[I_WOOD] >= 1 and near_table
        ok[MAKE_STONE_SWORD] &= inv[I_SSWORD] < 1 and inv[I_WOOD] >= 1 and inv[I_STONE] >= 1 and near_table
        ok[MAKE_IRON_SWORD] &= (inv[I_ISWORD] < 1 and inv[I_WOOD] >= 1 and inv[I_STONE] >= 1
                                and inv[I_COAL] >= 1 and inv[I_IRON] >= 1 and near_table and near_furn)
    logits[~ok] = -1e9
    p = np.exp(logits - logits.max())
    return int(s.rng.choice(17, p=p / p.sum()))


# ----------------------------------------------------------------------------- main
def choose_mode(c):
    """Return (priority, routine-or-None). Lower priority number = more urgent."""
    def free(name):
        return s.t >= s.giveup_until.get(name, -1)

    tree_done = c.inv[I_IPICK] >= 1 and (c.inv[I_ISWORD] >= 1 or s.t >= EARLY_STEPS)
    water_close = (c.blocks[CR - 2:CR + 3, CC - 2:CC + 3] == WATER).any()
    food_close = c.mobs[:, :, COW].any() or (c.blocks == RIPE).any()
    dusk = c.light < 0.5 and c.light < s.prev_light
    fed = c.drink > 3 and c.food > 3
    early = s.t < EARLY_STEPS and not tree_done  # let the network run its tech tree first
    if early:  # intervene only when the agent is about to die, and only for the cause
        if EARLY_MODE == "none":
            pass
        elif EARLY_MODE == "hp":
            if c.health <= 4:
                if FLAGS["drink"] and free("drink") and c.drink <= 3:
                    return 1, Drink()
                if FLAGS["eat"] and free("eat") and c.food <= 3:
                    return 2, Eat()
                if FLAGS["shelter"] and free("shelter") and c.energy <= 3 and not hostile_adjacent(c):
                    return 0, Shelter()
        else:
            if FLAGS["shelter"] and free("shelter") and (c.energy <= 1 or (c.health <= 3 and fed and not hostile_adjacent(c))):
                return 0, Shelter()
            if FLAGS["drink"] and free("drink") and c.drink <= 2:
                return 1, Drink()
            if FLAGS["eat"] and free("eat") and c.food <= 2:
                return 2, Eat()
        if FLAGS["topup"] and FLAGS["drink"] and free("drink") and c.drink <= 5 and water_close:
            return 3, Drink()  # cheap top-up when water is right here
        if FLAGS["base"] and free("base") and s.return_to_base and not (c.blocks == TABLE).any() and (s.mem == TABLE).any():
            return 4, Base()  # after a detour, bring the memory-less network back to its table
        return 9, None
    if FLAGS["shelter"] and free("shelter") and (
        c.energy <= 3 or (c.health <= 5 and fed and not hostile_adjacent(c)) or dusk
    ):
        return 0, Shelter()
    if FLAGS["drink"] and free("drink") and c.drink <= 3:
        return 1, Drink()
    if FLAGS["eat"] and free("eat") and c.food <= 3:
        return 2, Eat()
    if FLAGS["drink"] and free("drink") and c.drink <= 6 and water_close:
        return 3, Drink()
    if FLAGS["eat"] and free("eat") and c.food <= 6 and food_close:
        return 3, Eat()
    strong = c.health >= 5 and c.energy >= 3 and c.drink >= 5 and c.food >= 4 and c.light > 0.3
    if FLAGS["base"] and free("base") and s.return_to_base and not tree_done and not (c.blocks == TABLE).any() and (s.mem == TABLE).any():
        return 4, Base()
    if FLAGS["plant"] and free("plant") and strong and tree_done and s.plant is not None and not s.plant_eaten and s.t - s.plant[2] >= 600:
        return 4, EatPlant()
    if FLAGS["diamond"] and free("diamond") and strong and not s.diamond_done and tree_done:
        return 5, Diamond()
    return 9, None


def track_events(c, action):
    """Remember our sapling, cows and diamonds, using what we see and are about to do."""
    cows = np.nonzero(c.mobs[:, :, COW])
    if len(cows[0]):
        s.cow_seen = (*world(int(cows[0][0]), int(cows[1][0])), s.t)
    fr, fc = DIRS[c.facing]
    if action == PLACE_PLANT and c.inv[I_SAPLING] >= 1 and int(c.blocks[CR + fr, CC + fc]) == GRASS:
        if s.plant is None:
            s.plant = (*world(CR + fr, CC + fc), s.t)
    if action == DO and int(c.blocks[CR + fr, CC + fc]) == RIPE:
        if s.plant is not None and world(CR + fr, CC + fc) == s.plant[:2]:
            s.plant_eaten = True
    if action == DO and int(c.blocks[CR + fr, CC + fc]) == DIAMOND and c.inv[I_IPICK] >= 1:
        s.diamond_done = True
    if c.inv[I_DIAMOND] >= 1:
        s.diamond_done = True


def act(obs):
    c = parse(obs)
    update_position(c)
    action = None
    if c.sleeping:
        action = NOOP
    else:
        if s.mode is not None and (s.mode.done(c) or s.mode.age > 600):
            if isinstance(s.mode, (Drink, Eat, Shelter)):
                s.return_to_base = True  # the network may now be far from its table
            elif isinstance(s.mode, Base):
                s.return_to_base = False
            s.mode = None
        if isinstance(s.mode, (Diamond, EatPlant)) and c.health < 5:
            s.mode = None
        prio, cand = choose_mode(c)
        if s.mode is None:
            s.mode, s.prio = cand, prio
        elif cand is not None and prio < s.prio and not (isinstance(s.mode, Shelter) and s.mode.stage >= 1):
            s.mode, s.prio = cand, prio  # a more urgent need preempts
        if FLAGS["fight_adjacent"] and hostile_near(c, 2) and not (
            isinstance(s.mode, Shelter) and s.mode.stage >= 7
        ):
            action = masked_sample(c)
        elif s.mode is not None:
            action = s.mode.step(c)
            if action is None:
                s.giveup_until[s.mode.name] = s.t + s.mode.cooldown
                if isinstance(s.mode, Base):
                    s.return_to_base = False
                s.mode = None
        if action is None:
            action = masked_sample(c)
    track_events(c, action)
    s.prev_light = c.light
    s.last_action, s.last_blocks, s.last_mobs = action, c.blocks, c.mobs
    s.t += 1
    return int(action)

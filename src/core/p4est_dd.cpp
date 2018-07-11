#include "p4est_dd.hpp"
//--------------------------------------------------------------------------------------------------
#ifdef DD_P4EST
//--------------------------------------------------------------------------------------------------
#ifdef LEES_EDWARDS
#error "p4est and Lees-Edwards are not compatible yet."
#endif
//--------------------------------------------------------------------------------------------------
#include <p4est_to_p8est.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_extended.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#include "Cell.hpp"
#include "domain_decomposition.hpp"
#include "ghosts.hpp"
#include "p4est_utils.hpp"
#include "repart.hpp"
#include "utils/serialization/ParticleList.hpp"

//#include <boost/mpi/nonblocking.hpp>


// For intrinsics version of cell_morton_idx
#ifdef __BMI2__
#include <x86intrin.h>
#endif

enum class CellType {
  Inner,
  Boundary,
  Ghost,
};

// Structure that stores basic grid information
struct CellInfo {
  uint64_t idx; // a unique index within all cells (as used by p4est for locals)
  int rank; // the rank of this cell (equals this_node for locals)
  CellType type; // shell information (0: inner local cell, 1: boundary local cell, 2: ghost cell)
  int neighbor[26]; // unique index of the fullshell neighborhood cells (as in p4est)
                    // Might be "-1" if searching for neighbors over a non-periodic boundary.
  std::array<int, 3> coord; // cartesian coordinates of the cell
};

// Structure to store communications
struct comm_t {
  int rank; // Rank of the communication partner
  int cnt;  // number of cells to communicate
  int dir;  // Bitmask for communication direction. MSB ... z_r,z_l,y_r,y_l,x_r,x_l LSB
  std::vector<int> idx; // List of cell indexes
};


namespace {
// Needs to be a class because "p4est" needs to be destructed before "p4est_conn"!
// Destruction order for non-static members is guaranteed to be in the order of declaration.
struct __DS {
  castable_unique_ptr<p4est_connectivity_t> p4est_conn;
  castable_unique_ptr<p4est_t> p4est;
  std::vector<CellInfo> p4est_cellinfo;
};

static __DS ds;
}

//--------------------------------------------------------------------------------------------------
// Creates the irregular DD using p4est
static void dd_p4est_create_grid (bool isRepart = false);
// Compute communication partners for this node and fill internal lists
static void dd_p4est_comm ();
// Mark all cells either local or ghost. Local cells are arranged before ghost cells
static void dd_p4est_mark_cells ();
// Prepare a GhostCommunicator using the internal lists
static void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part);

// Fill the IA_NeighborList and compute the half-shell for p4est based DD
static void dd_p4est_init_cell_interactions();

// Map a position to a cell, returns NULL if not in local (+ ROUND_ERR_PREC*boxl) domain
static Cell* dd_p4est_save_position_to_cell(double pos[3]);
// Map a position to a cell, returns NULL if not in local domain
static Cell* dd_p4est_position_to_cell(double pos[3]);
//void dd_p4est_position_to_cell(double pos[3], int* idx);
Cell* dd_p4est_position_to_cell_strict(double pos[3]);

// Map a position to a global processor index
static int dd_p4est_pos_to_proc(double pos[3]);
// Compute a Morton index for a position (this is not equal to the p4est index)
static uint64_t dd_p4est_pos_morton_idx(double pos[3]);

/** Revert the order of a communicator: After calling this the
    communicator is working in reverted order with exchanged
    communication types GHOST_SEND <-> GHOST_RECV. */
static void dd_p4est_revert_comm_order (GhostCommunicator *comm);

static void dd_p4est_init_internal_minimal(p4est_ghost_t *p4est_ghost, p4est_mesh_t *p4est_mesh);
//--------------------------------------------------------------------------------------------------
static int brick_size[3]; // number of trees in each direction
static int grid_size[3];  // number of quadrants/cells in each direction
static std::vector<int> first_morton_idx_of_rank; // n_nodes + 1 elements, storing the first Morton index of local cell
static int grid_level = 0;
//--------------------------------------------------------------------------------------------------
static std::vector<comm_t> comm_send;  // Internal send lists, idx maps to local cells
static std::vector<comm_t> comm_recv;  // Internal recv lists, idx maps to ghost cells
static int num_comm_send = 0;     // Number of send lists
static int num_comm_recv = 0;     // Number of recc lists (should be euqal to send lists)
static std::vector<int> comm_proc;     // Number of communicators per Process
static std::vector<int> comm_rank;     // Communication partner index for a certain communication
static int num_comm_proc = 0;     // Total Number of bidirectional communications
//--------------------------------------------------------------------------------------------------
static std::vector<p4est_gloidx_t> old_global_first_quadrant; // Old global_first_quadrants of the p4est before repartitioning.
//--------------------------------------------------------------------------------------------------
static size_t num_cells = 0;
static size_t num_local_cells = 0;
static size_t num_ghost_cells = 0;
//--------------------------------------------------------------------------------------------------
static const int neighbor_lut[3][3][3] = { // Mapping from [z][y][x] to p4est neighbor index
  {{18, 6,19},{10, 4,11},{20, 7,21}},
  {{14, 2,15},{ 0,-1, 1},{16, 3,17}},  //[1][1][1] is the cell itself
  {{22, 8,23},{12, 5,13},{24, 9,25}}
};
//--------------------------------------------------------------------------------------------------
static const int half_neighbor_idx[14] = { // p4est neighbor index of halfshell [0] is cell itself
  -1, 1, 16, 3, 17, 22, 8, 23, 12, 5, 13, 24, 9, 25
};
//--------------------------------------------------------------------------------------------------
// Datastructure for partitioning holding the number of quads per process
// Empty as long as tclcommand_repart has not been called.
static std::vector<p4est_locidx_t> part_nquads;
//--------------------------------------------------------------------------------------------------

enum class Direction {
  Left,
  Right
};

static bool is_on_boundary(const CellInfo& ci, int dim, Direction lr) {
  if (lr == Direction::Left)
    return ci.coord[dim] == 0;
  else
    return ci.coord[dim] == grid_size[dim] - 1;
}

static bool is_on_boundary(const CellInfo& ci) {
  for (int d = 0; d < 3; ++d) {
    if (is_on_boundary(ci, d, Direction::Left)
        || is_on_boundary(ci, d, Direction::Right))
      return true;
  }
  return false;
}

struct CommunicationStatus {
  enum class ReceiveStatus {
    RECV_NOTSTARTED,
    RECV_COUNT,
    RECV_PARTICLES,
    RECV_DYNDATA,
    RECV_DONE
  };

  CommunicationStatus(int ncomm)
      : nextstatus(ncomm, ReceiveStatus::RECV_COUNT),
        donestatus(ncomm, ReceiveStatus::RECV_NOTSTARTED)
  {
  }

  ReceiveStatus expected(int idx)
  {
    if (nextstatus[idx] <= donestatus[idx]) {
      std::cerr << "[" << this_node << "] "
                << "Error in communication sequence: "
                << "Expecting status " << static_cast<int>(nextstatus[idx])
                << " to happen but is already done ("
                << static_cast<int>(donestatus[idx]) << ")" << std::endl;
      errexit();
    }

    incr_recvstatus(donestatus[idx]);
    return nextstatus[idx];
  }

  void next(int idx) { incr_recvstatus(nextstatus[idx]); }

  static void incr_recvstatus(ReceiveStatus &r)
  {
    if (r == ReceiveStatus::RECV_DONE) {
      std::cerr << "[" << this_node << "] "
                << "Trying to increment bad communication status: "
                << static_cast<int>(r) << std::endl;
    }
    r = static_cast<ReceiveStatus>(static_cast<int>(r) + 1);
  }

 private:
  std::vector<ReceiveStatus> nextstatus;
  std::vector<ReceiveStatus> donestatus;
};

static inline void DIE_IF_TAG_MISMATCH(int act, int desired, const char *str) {
  if (act != desired) {
    std::cerr << "[" << this_node << "] TAG MISMATCH!"
              << "Desired: " << desired << " Got: " << act << std::endl;
    errexit();
  }
}

static const int LOC_EX_CNT_TAG  = 11011;
static const int LOC_EX_PART_TAG = 11022;
static const int LOC_EX_DYN_TAG  = 11033;
static const int GLO_EX_CNT_TAG  = 22011;
static const int GLO_EX_PART_TAG = 22022;
static const int GLO_EX_DYN_TAG  = 22033;
static const int REP_EX_CNT_TAG  = 33011;
static const int REP_EX_PART_TAG = 33022;
static const int REP_EX_DYN_TAG  = 33033;

static const int COMM_RANK_NONE  = 999999;
//--------------------------------------------------------------------------------------------------

p4est_t* dd_p4est_get_p4est() {
  return ds.p4est;
}

int dd_p4est_get_n_trees(int dir) {
  return brick_size[dir];
}

// Returns the morton index for given cartesian coordinates.
// Note: This is not the index of the p4est quadrants. But the ordering is the same.
uint64_t dd_p4est_cell_morton_idx(int x, int y, int z) {
#ifdef __BMI2__
  using pdep_uint = unsigned long long;
  static constexpr pdep_uint mask_x = 0x1249249249249249;
  static constexpr pdep_uint mask_y = 0x2492492492492492;
  static constexpr pdep_uint mask_z = 0x4924924924924924;

  return _pdep_u64(static_cast<pdep_uint>(x), mask_x)
           | _pdep_u64(static_cast<pdep_uint>(y), mask_y)
           | _pdep_u64(static_cast<pdep_uint>(z), mask_z);
#else
  uint64_t idx = 0;
  uint64_t bit = 1;

  for (int i = 0; i < 21; ++i) {
    if ((x & 1))
      idx |= bit;
    x >>= 1;
    bit <<= 1;
    if ((y & 1))
      idx |= bit;
    y >>= 1;
    bit <<= 1;
    if ((z & 1))
      idx |= bit;
    z >>= 1;
    bit <<= 1;
  }

  return idx;
#endif
}
//--------------------------------------------------------------------------------------------------
int dd_p4est_full_shell_neigh(int cell, int neighidx)
{
    if (neighidx >= 0 && neighidx < 26)
        return ds.p4est_cellinfo[cell].neighbor[neighidx];
    else if (neighidx == 26)
        return cell;
    else {
        fprintf(stderr, "dd_p4est_full_shell_neigh: Require 0 <= neighidx < 27.\n");
        errexit();
    }
    // Not reached.
    return 0;
}
//--------------------------------------------------------------------------------------------------
static inline int count_trailing_zeros(int x)
{
  int z = 0;
  for (; (x & 1) == 0; x >>= 1) z++;
  return z;
}

// Compute the grid- and bricksize according to box_l and maxrange
void dd_p4est_set_cellsize_optimal() {
  int ncells[3] = {1, 1, 1};

  // compute number of cells
  if (max_range > ROUND_ERROR_PREC * box_l[0]) {
    ncells[0] = std::max<int>(box_l[0] / max_range, 1);
    ncells[1] = std::max<int>(box_l[1] / max_range, 1);
    ncells[2] = std::max<int>(box_l[2] / max_range, 1);
  }

  grid_size[0] = ncells[0];
  grid_size[1] = ncells[1];
  grid_size[2] = ncells[2];

  // divide all dimensions by biggest common power of 2
  grid_level = count_trailing_zeros(ncells[0] | ncells[1] | ncells[2]);

  brick_size[0] = ncells[0] >> grid_level;
  brick_size[1] = ncells[1] >> grid_level;
  brick_size[2] = ncells[2] >> grid_level;

  for (int i = 0; i < 3; ++i) {
    dd.cell_size[i] = box_l[i] / static_cast<double>(grid_size[i]);
    dd.inv_cell_size[i] = 1.0 / dd.cell_size[i];
  }
  max_skin = std::min(std::min(dd.cell_size[0], dd.cell_size[1]), dd.cell_size[2]) - max_cut;
}
//--------------------------------------------------------------------------------------------------
// Reinitializes all grid parameters, i.e. grid sizes and p4est structs.
static void dd_p4est_initialize_grid() {
  // Set global variables
  dd_p4est_set_cellsize_optimal(); // Sets grid_size and brick_size

  // Create p4est structs
  auto oldconn = std::move(ds.p4est_conn);
  ds.p4est_conn =
      std::unique_ptr<p4est_connectivity_t>(p8est_connectivity_new_brick(
          brick_size[0], brick_size[1], brick_size[2], PERIODIC(0),
          PERIODIC(1), PERIODIC(2)));
  ds.p4est = std::unique_ptr<p4est_t>(
      p4est_new_ext(comm_cart, ds.p4est_conn, 0, grid_level, true,
                    static_cast<size_t>(0), nullptr, nullptr));
}

/** Returns the global morton idx of a quad. 
 */
static uint64_t dd_p4est_quad_global_morton_idx(p4est_quadrant_t &q)
{
  std::array<double, 3> xyz;
  int scal = 1 << grid_level;
  p4est_qcoord_to_vertex(ds.p4est_conn, q.p.which_tree, q.x, q.y, q.z, xyz.data());
  return dd_p4est_cell_morton_idx(xyz[0] * scal, xyz[1] * scal, xyz[2] * scal);
}

/** Returns the maximum global(!) morton index for a grid of level
 * grid_level.
 */
static uint64_t dd_p4est_grid_max_global_morton_idx()
{
  int i = 1 << grid_level;
  while(i < grid_size[0] || i < grid_size[1] || i < grid_size[2])
    i <<= 1;
  auto ui = static_cast<uint64_t>(i);
  return ui * ui * ui;
}

void dd_p4est_create_grid(bool isRepart) {
  // Note: In case of LB_ADAPTIVE (lb_adaptive_is_defined), calling
  // p4est_partition is handled by lbmd_repart.[ch]pp. This means that must not
  // do anything related torepartitioning in this case.

  // Clear data to prevent accidental use of old stuff
  comm_rank.clear();
  comm_proc.clear();
  comm_recv.clear();
  comm_send.clear();
  first_morton_idx_of_rank.clear();
  ds.p4est_cellinfo.clear();

  if (!isRepart) 
    dd_p4est_initialize_grid();
#if !defined(LB_ADAPTIVE)
  else
    p4est_dd_repart_preprocessing(); // In case of LB_ADAPTIVE, lbmd_repart does this
#endif

  // Repartition uniformly if part_nquads is empty (because not repart has been
  // done yet). Else use part_nquads as given partitioning.
  // If LB_ADAPTIVE is defined, do not repartition here at all, since we handle
  // this case in lbmd_repart.[ch]pp.
#if !defined(LB_ADAPTIVE)
    if (part_nquads.size() == 0)
      p4est_partition(ds.p4est, 0, nullptr);
    else
      p4est_partition_given(ds.p4est, part_nquads.data());
#endif

  auto p4est_ghost = castable_unique_ptr<p4est_ghost_t>(
      p4est_ghost_new(ds.p4est, P8EST_CONNECT_CORNER));
  auto p4est_mesh = castable_unique_ptr<p4est_mesh_t>(
      p4est_mesh_new_ext(ds.p4est, p4est_ghost, 1, 1, 0, P8EST_CONNECT_CORNER));


  // create space filling inforamtion about first quads per node from p4est
  first_morton_idx_of_rank.resize(n_nodes + 1);
  for (int i = 0; i < n_nodes; ++i)
    first_morton_idx_of_rank[i] = dd_p4est_quad_global_morton_idx(ds.p4est->global_first_position[i]);

  first_morton_idx_of_rank[n_nodes] = dd_p4est_grid_max_global_morton_idx();

  dd_p4est_init_internal_minimal(p4est_ghost, p4est_mesh);

  // allocate memory
  realloc_cells(num_cells);
  realloc_cellplist(&local_cells, local_cells.n = num_local_cells);
  realloc_cellplist(&ghost_cells, ghost_cells.n = num_ghost_cells);

  // Fill internal communator lists
  dd_p4est_comm();

  p4est_utils_init();
}
//--------------------------------------------------------------------------------------------------
/** Returns the morton idx of a quadrant local to tree of index "treeidx".
 */
static std::array<int, 3> morton_idx_of_quadrant(p4est_quadrant_t *q, p4est_topidx_t treeidx) {
  std::array<double, 3> xyz;
  p4est_qcoord_to_vertex(ds.p4est_conn, treeidx, q->x, q->y, q->z, xyz.data());
  int scal = 1 << p4est_tree_array_index(ds.p4est->trees, treeidx)->maxlevel;

  std::array<int, 3> coord;
  for (int d = 0; d < 3; ++d)
    coord[d] = xyz[d] * scal;
  return coord;
}

//--------------------------------------------------------------------------------------------------
void dd_p4est_init_internal_minimal (p4est_ghost_t *p4est_ghost, p4est_mesh_t *p4est_mesh) {
  num_local_cells = (size_t) ds.p4est->local_num_quadrants;
  num_ghost_cells = (size_t) p4est_ghost->ghosts.elem_count;
  num_cells = num_local_cells + num_ghost_cells;
  castable_unique_ptr<sc_array_t> ni = sc_array_new(sizeof(int));

  ds.p4est_cellinfo.resize(num_cells);

  for (int i = 0; i < num_local_cells; ++i) {
    ds.p4est_cellinfo[i].idx = i;
    ds.p4est_cellinfo[i].rank = this_node;
    ds.p4est_cellinfo[i].coord = morton_idx_of_quadrant(p4est_mesh_get_quadrant(ds.p4est, p4est_mesh, i), p4est_mesh->quad_to_tree[i]);
    ds.p4est_cellinfo[i].type = is_on_boundary(ds.p4est_cellinfo[i])? CellType::Boundary: CellType::Inner;
    
    for (int n = 0; n < 26; ++n) {
      ds.p4est_cellinfo[i].neighbor[n] = -1;
      p4est_mesh_get_neighbors(ds.p4est, p4est_ghost, p4est_mesh, i, n, nullptr, nullptr, ni);
      if (ni->elem_count > 1) // more than 1 neighbor in this direction
        printf("%i %i %li strange stuff\n",i,n,ni->elem_count);
      if (ni->elem_count > 0) {
        int neighrank = *((int*)sc_array_index_int(ni,0));
        ds.p4est_cellinfo[i].neighbor[n] = neighrank;
        if (neighrank >= ds.p4est->local_num_quadrants) { // Ghost cell
          ds.p4est_cellinfo[i].type = CellType::Boundary;
        }
      }
      sc_array_truncate(ni);
    }
  }
  
  for (int g = 0; g < num_ghost_cells; ++g) {
    int i = num_local_cells + g;
    
    ds.p4est_cellinfo[i].idx = g;
    ds.p4est_cellinfo[i].rank = p4est_mesh->ghost_to_proc[g];
    ds.p4est_cellinfo[i].type = CellType::Ghost;
    p4est_quadrant_t *q = p4est_quadrant_array_index(&p4est_ghost->ghosts, g);
    ds.p4est_cellinfo[i].coord = morton_idx_of_quadrant(q, q->p.piggy3.which_tree);
    std::fill(std::begin(ds.p4est_cellinfo[i].neighbor), std::end(ds.p4est_cellinfo[i].neighbor), -1);
  }
}

// Compute communication partners and the cells that need to be communicated
void dd_p4est_comm () {
  // List of cell idx marked for send/recv for each process
  std::vector<std::vector<int>> send_idx(n_nodes);
  std::vector<std::vector<int>> recv_idx(n_nodes);
  
  comm_proc.resize(n_nodes);
  std::fill(std::begin(comm_proc), std::end(comm_proc), -1);

  for (int i = 0; i < num_cells; ++i) {
    // is ghost cell -> add to recv lists
    if (ds.p4est_cellinfo[i].rank >= 0 && ds.p4est_cellinfo[i].type == CellType::Ghost) {
      int nrank = ds.p4est_cellinfo[i].rank;
      recv_idx[nrank].push_back(i);
    }
    if (ds.p4est_cellinfo[i].type == CellType::Boundary) { // is mirror cell -> add to send lists
      // loop fullshell
      for (int n = 0; n < 26; ++n) {
        int nidx = ds.p4est_cellinfo[i].neighbor[n];
        if (nidx < 0)
          continue; // invalid neighbor
        int nrank = ds.p4est_cellinfo[nidx].rank;
        if (nrank < 0)
          continue; // invalid neighbor
        if (ds.p4est_cellinfo[nidx].type != CellType::Ghost)
          continue; // no need to send to local cell
        // add once for each rank
        if (send_idx[nrank].empty() || send_idx[nrank].back() != i)
          send_idx[nrank].push_back(i);
      }
    }
  }
  
  num_comm_proc = 0;
  for (int i=0; i<n_nodes; ++i) {
    if (send_idx[i].size() != 0 && recv_idx[i].size() != 0) {
      comm_proc[i] = num_comm_proc;
      num_comm_proc += 1;
    } else if (!(send_idx[i].size() == 0 && recv_idx[i].size() == 0)) {
      printf("[%i] : Unexpected mismatch in send and receive lists.\n", this_node);
      errexit();
    }
  }
  
  num_comm_recv = num_comm_proc;
  num_comm_send = num_comm_proc;
  comm_recv.resize(num_comm_recv);
  comm_send.resize(num_comm_send);
  comm_rank.resize(num_comm_proc, COMM_RANK_NONE);
  
  for (int n=0; n<n_nodes; ++n) {
    if (comm_proc[n] >= 0) {
      comm_rank[comm_proc[n]] = n;
      
      comm_recv[comm_proc[n]].cnt = recv_idx[n].size();
      comm_recv[comm_proc[n]].dir = 0;
      comm_recv[comm_proc[n]].rank = n;
      comm_recv[comm_proc[n]].idx = std::move(recv_idx[n]);

      comm_send[comm_proc[n]].cnt = send_idx[n].size();
      comm_send[comm_proc[n]].dir = 0;
      comm_send[comm_proc[n]].rank = n;
      comm_send[comm_proc[n]].idx = std::move(send_idx[n]);
    }
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part) {
  prepare_comm(comm, data_part, num_comm_send + num_comm_recv, true);
  int cnt = 0;
  for (int i=0;i<num_comm_send;++i) {
    comm->comm[cnt].type = GHOST_SEND;
    comm->comm[cnt].node = comm_send[i].rank;
    // The tag distinguishes communications to the same rank
    comm->comm[cnt].tag = comm_send[i].dir;
    comm->comm[cnt].part_lists = (Cell**)Utils::malloc(comm_send[i].cnt*sizeof(Cell*));
    comm->comm[cnt].n_part_lists = comm_send[i].cnt;
    for (int n=0;n<comm_send[i].cnt;++n) {
      comm->comm[cnt].part_lists[n] = &cells[comm_send[i].idx[n]];
    }
    ++cnt;
  }
  for (int i=0;i<num_comm_recv;++i) {
    comm->comm[cnt].type = GHOST_RECV;
    comm->comm[cnt].node = comm_recv[i].rank;
    comm->comm[cnt].tag = comm_recv[i].dir; // The tag is the same as above
    // But invert x, y and z direction now. Direction from where the data came
    if ((comm_recv[i].dir &  3)) comm->comm[cnt].tag ^=  3;
    if ((comm_recv[i].dir & 12)) comm->comm[cnt].tag ^= 12;
    if ((comm_recv[i].dir & 48)) comm->comm[cnt].tag ^= 48;
    comm->comm[cnt].part_lists = (Cell**)Utils::malloc(comm_recv[i].cnt*sizeof(Cell*));
    comm->comm[cnt].n_part_lists = comm_recv[i].cnt;
    for (int n=0;n<comm_recv[i].cnt;++n) {
      comm->comm[cnt].part_lists[n] = &cells[comm_recv[i].idx[n]];
    }
    ++cnt;
  }

}
//--------------------------------------------------------------------------------------------------
void dd_p4est_revert_comm_order (GhostCommunicator *comm) {
  /* exchange SEND/RECV */
  for(int i = 0; i < comm->num; i++) {
    if (comm->comm[i].type == GHOST_SEND)
      comm->comm[i].type = GHOST_RECV;
    else if (comm->comm[i].type == GHOST_RECV)
      comm->comm[i].type = GHOST_SEND;
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_mark_cells () {
  // Take the memory map as they are. First local cells (along Morton curve).
  // This also means cells has the same ordering as p4est_cellinfo
  for (int c=0;c<num_local_cells;++c)
    local_cells.cell[c] = &cells[c];
  // Ghost cells come after local cells
  for (int c=0;c<num_ghost_cells;++c)
    ghost_cells.cell[c] = &cells[num_local_cells + c];
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_init_cell_interactions() {
  for (int i = 0; i < local_cells.n; ++i) {
    cells[i].m_neighbors.clear();
    cells[i].m_neighbors.reserve(14);
    for (int n = 1; n < 14; ++n) {
      auto neighidx = ds.p4est_cellinfo[i].neighbor[half_neighbor_idx[n]];
      // Check for invalid cells
      if (neighidx >= 0 && neighidx < num_cells)
        local_cells.cell[i]->m_neighbors.emplace_back(&cells[neighidx]);
    }
  }
}
//--------------------------------------------------------------------------------------------------
Cell* dd_p4est_position_to_cell_strict(double pos[3]) {
  // Does the same as dd_p4est_save_position_to_cell but does not extend the local domain
  // by the error bounds

  auto shellidxcomp = [](const CellInfo& s, uint64_t idx) {
    uint64_t sidx = dd_p4est_cell_morton_idx(s.coord[0],
                                             s.coord[1],
                                             s.coord[2]);
    return sidx < idx;
  };

  const auto needle = dd_p4est_pos_morton_idx(pos);

  auto local_end = std::begin(ds.p4est_cellinfo) + num_local_cells;
  auto it = std::lower_bound(std::begin(ds.p4est_cellinfo),
                             local_end,
                             needle,
                             shellidxcomp);
  if (it != local_end
        && dd_p4est_cell_morton_idx(it->coord[0],
                                    it->coord[1],
                                    it->coord[2]) == needle)
    return &cells[std::distance(std::begin(ds.p4est_cellinfo), it)];
  else
    return nullptr;
}
//--------------------------------------------------------------------------------------------------
Cell* dd_p4est_save_position_to_cell(double pos[3]) {
  Cell *c;
  // This function implicitly uses binary search by using
  // dd_p4est_position_to_cell

  if ((c = dd_p4est_position_to_cell_strict(pos)))
    return c;

  // If pos is outside of the local domain try the bounding box enlarged
  // by ROUND_ERROR_PREC
  for (int i = -1; i <= 1; i += 2) {
    for (int j = -1; j <= 1; j += 2) {
      for (int k = -1; k <= 1; k += 2) {
        double spos[3] = { pos[0] + i * box_l[0] * ROUND_ERROR_PREC,
                           pos[1] + j * box_l[1] * ROUND_ERROR_PREC,
                           pos[2] + k * box_l[2] * ROUND_ERROR_PREC };
        if ((c = dd_p4est_position_to_cell_strict(spos)))
          return c;
      }
    }
  }

  return nullptr;
}
//--------------------------------------------------------------------------------------------------
Cell* dd_p4est_position_to_cell(double pos[3]) {
  // Accept OOB particles for the sake of IO.
  // If this is used, you need to manually do a global(!) resort afterwards
  // MPI-IO does so.
  Cell *c = dd_p4est_position_to_cell_strict(pos);

  if (c) {
    return c;
  } else {
    return &cells[0];
  }
}
//--------------------------------------------------------------------------------------------------
static bool is_out_of_box(Particle& p, const std::array<int, 3> &off) {
  for (int d = 0; d < 3; ++d)
    if (off[d] != 1 && (p.r.p[d] < 0.0 || p.r.p[d] >= box_l[d]))
      return true;
  return false;
}
//--------------------------------------------------------------------------------------------------
// Checks all particles and resorts them to local cells or sendbuffers
// Note: Particle that stay local and are moved to a cell with higher index are touched twice.
// This is not the most efficient way! It would be better to first remember the cell-particle
// index pair of those particles in a vector and move them at the end.
void dd_p4est_fill_sendbuf (ParticleList *sendbuf, std::vector<int> *sendbuf_dyn)
{
  std::array<double, 3> cell_lc, cell_hc, ext_cell_lc, ext_cell_hc;
  // Loop over all cells and particles
  for (int i=0;i<num_local_cells;++i) {
    Cell* cell = local_cells.cell[i];
    CellInfo* shell = &ds.p4est_cellinfo[i];

    for (int d = 0; d < 3; ++d) {
      cell_lc[d] = dd.cell_size[d] * static_cast<double>(shell->coord[d]);
      cell_hc[d] = cell_lc[d] + dd.cell_size[d];

      ext_cell_lc[d] = cell_lc[d];
      ext_cell_hc[d] = cell_hc[d];
      if (is_on_boundary(*shell, d, Direction::Left))
        ext_cell_lc[d] -= 0.5 * ROUND_ERROR_PREC*box_l[d];
      if (is_on_boundary(*shell, d, Direction::Right))
        ext_cell_hc[d] += 0.5 * ROUND_ERROR_PREC*box_l[d];
    }

    for (int p=0;p<cell->n;++p) {
      Particle* part = &cell->part[p];
      std::array<int, 3> off;

      // Check if particle has left the cell. (The local domain is extended by half round error)
      for (int d = 0; d < 3; ++d) {
        if (part->r.p[d] < ext_cell_lc[d])
          off[d] = 0;
        else if (part->r.p[d] >= ext_cell_hc[d])
          off[d] = 2;
        else
          off[d] = 1;
      }

      int nidx = neighbor_lut[off[2]][off[1]][off[0]];
      if (nidx != -1) { // Particle p outside of cell i
        // recalculate neighboring cell to prevent rounding errors
        // If particle left local domain, check for correct ghost cell, thus without ROUND_ERROR_PREC
        for (int d = 0; d < 3; ++d) {
          if (part->r.p[d] < cell_lc[d])
            off[d] = 0;
          else if (part->r.p[d] >= cell_hc[d])
            off[d] = 2;
          else
            off[d] = 1;
        }

        // get neighbor cell
        nidx = shell->neighbor[neighbor_lut[off[2]][off[1]][off[0]]];
        if (nidx >= num_local_cells) { // Remote Cell (0:num_local_cells-1) -> local, other: ghost
          if (ds.p4est_cellinfo[nidx].rank >= 0) { // This ghost cell is linked to a process
            // With minimal ghost this condition is always true
            if (ds.p4est_cellinfo[nidx].rank != this_node) { // It is a remote process
              // copy data to sendbuf according to rank
              int li = comm_proc[ds.p4est_cellinfo[nidx].rank];
              sendbuf_dyn[li].insert(sendbuf_dyn[li].end(), part->bl.e, part->bl.e + part->bl.n);
#ifdef EXCLUSIONS
              sendbuf_dyn[li].insert(sendbuf_dyn[li].end(), part->el.e, part->el.e + part->el.n);
#endif
              int pid = part->p.identity;
              //fold_position(part->r.p, part->l.i);
              move_indexed_particle(&sendbuf[li], cell, p);
              local_particles[pid] = nullptr;
              if(p < cell->n) p -= 1;
            } else { // particle stays local, but since it went to a ghost it has to be folded
              fold_position(part->r.p, part->l.i);
              move_indexed_particle(&cells[ds.p4est_cellinfo[nidx].idx], cell, p);
              if(p < cell->n) p -= 1;
            }
          } else { // Particle left global domain and is not tracked by any process anymore
            runtimeErrorMsg() << "particle " << p << " on process " << this_node << " is OB";
            fprintf(stderr, "%i : part %i cell %i is OB [%lf %lf %lf]\n", this_node, i, p, part->r.p[0], part->r.p[1], part->r.p[2]);
          }
        } else { // Local Cell, just move the partilce
          if (is_out_of_box(*part, off)) { // Local periodic boundary
            fold_position(part->r.p, part->l.i);
          }
          move_indexed_particle(&cells[nidx], cell, p);
          if(p < cell->n) p -= 1;
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
static int dd_async_exchange_insert_particles(ParticleList *recvbuf, int global_flag, int from) {
  int dynsiz = 0;
  update_local_particles(recvbuf);

  for (int p = 0; p < recvbuf->n; ++p) {
    double op[3] = {recvbuf->part[p].r.p[0], recvbuf->part[p].r.p[1], recvbuf->part[p].r.p[2]};
    // Sender folds in global case to get the correct receiver rank.
    // Do not undo this folding here.
    //if (!global_flag)
      fold_position(recvbuf->part[p].r.p, recvbuf->part[p].l.i);

    dynsiz += recvbuf->part[p].bl.n;
#ifdef EXCLUSIONS
    dynsiz += recvbuf->part[p].el.n;
#endif

    Cell* target = dd_p4est_save_position_to_cell(recvbuf->part[p].r.p.data());
    if (target) {
      append_indexed_particle(target, std::move(recvbuf->part[p]));
    } else {
      fprintf(stderr, "proc %i received remote particle p %i out of domain, global %i from proc %i\n\t%lfx%lfx%lf, glob morton idx %li, pos2proc %i\n\told pos %lfx%lfx%lf\n",
        this_node, recvbuf->part[p].p.identity, global_flag, from,
        recvbuf->part[p].r.p[0], recvbuf->part[p].r.p[1], recvbuf->part[p].r.p[2],
        dd_p4est_pos_morton_idx(recvbuf->part[p].r.p.data()),
               dd_p4est_pos_to_proc(recvbuf->part[p].r.p.data()),
               op[0], op[1], op[2]);
      errexit();
    }
  }
  return dynsiz;
}
//--------------------------------------------------------------------------------------------------
static void dd_async_exchange_insert_dyndata(ParticleList *recvbuf, std::vector<int> &dynrecv) {
  int read = 0;

  for (int pc = 0; pc < recvbuf->n; pc++) {
    // Use local_particles to find the correct particle address since the
    // particles from recvbuf have already been copied by dd_append_particles
    // in dd_async_exchange_insert_particles.
    Particle *p = local_particles[recvbuf->part[pc].p.identity];
    if (p->bl.n > 0) {
      if (!(p->bl.e = (int *) malloc(p->bl.max * sizeof(int)))) {
        fprintf(stderr, "Tod.\n");
        errexit();
      }
      memcpy(p->bl.e, &dynrecv[read], p->bl.n * sizeof(int));
      read += p->bl.n;
    } else {
      p->bl.e = nullptr;
      p->bl.max = 0;
    }
#ifdef EXCLUSIONS
    if (p->el.n > 0) {
      if (!(p->el.e = (int *) malloc(p->el.max * sizeof(int)))) {
        fprintf(stderr, "Tod.\n");
        errexit();
      }
      memcpy(p->el.e, &dynrecv[read], p->el.n*sizeof(int));
      read += p->el.n;
    }
    else {
      p->el.e = nullptr;
      p->el.max = 0;
    }
#endif
  }
}
void dd_p4est_local_exchange_and_sort_particles() {
  // Prepare all send and recv buffers to all neighboring processes
  std::vector<ParticleList> sendbuf(num_comm_proc), recvbuf(num_comm_proc);
  std::vector<std::vector<int>> sendbuf_dyn(num_comm_proc),
      recvbuf_dyn(num_comm_proc);
  std::vector<MPI_Request> sreq(3 * num_comm_proc, MPI_REQUEST_NULL);
  std::vector<MPI_Request> rreq(num_comm_proc, MPI_REQUEST_NULL);
  std::vector<int> nrecvpart(num_comm_proc, 0);

  for (int i = 0; i < num_comm_proc; ++i) {
    init_particlelist(&sendbuf[i]);
    init_particlelist(&recvbuf[i]);
    // Invoke receive for the number of particles to be received from
    // all neighbors
    MPI_Irecv(&nrecvpart[i], 1, MPI_INT, comm_rank[i], LOC_EX_CNT_TAG,
              comm_cart, &rreq[i]);
  }

  // Fill the send buffers with particles that leave the local domain
  dd_p4est_fill_sendbuf(sendbuf.data(), sendbuf_dyn.data());

  // send number of particles, particles, and particle data
  for (int i = 0; i < num_comm_proc; ++i) {
    MPI_Isend(&sendbuf[i].n, 1, MPI_INT, comm_rank[i], LOC_EX_CNT_TAG,
              comm_cart, &sreq[i]);
    if (sendbuf[i].n <= 0)
      continue;
    MPI_Isend(sendbuf[i].part, sendbuf[i].n * sizeof(Particle), MPI_BYTE,
              comm_rank[i], LOC_EX_PART_TAG, comm_cart,
              &sreq[i + num_comm_proc]);
    if (sendbuf_dyn[i].size() <= 0)
      continue;
    MPI_Isend(sendbuf_dyn[i].data(), sendbuf_dyn[i].size(), MPI_INT,
              comm_rank[i], LOC_EX_DYN_TAG, comm_cart,
              &sreq[i + 2 * num_comm_proc]);
  }

  // Receive all data
  MPI_Status status;
  CommunicationStatus commstat(num_comm_proc);

  while (true) {
    int recvidx;
    MPI_Waitany(num_comm_proc, rreq.data(), &recvidx, &status);
    if (recvidx == MPI_UNDEFINED)
      break;
    int dyndatasiz, source = status.MPI_SOURCE, tag = status.MPI_TAG;

    switch (commstat.expected(recvidx)) {
    case CommunicationStatus::ReceiveStatus::RECV_COUNT:
      DIE_IF_TAG_MISMATCH(tag, LOC_EX_CNT_TAG, "Local exchange count");
      if (nrecvpart[recvidx] > 0) {
        realloc_particlelist(&recvbuf[recvidx], nrecvpart[recvidx]);
        MPI_Irecv(recvbuf[recvidx].part, nrecvpart[recvidx] * sizeof(Particle),
                  MPI_BYTE, source, LOC_EX_PART_TAG, comm_cart, &rreq[recvidx]);
        commstat.next(recvidx);
      }
      break;
    case CommunicationStatus::ReceiveStatus::RECV_PARTICLES:
      DIE_IF_TAG_MISMATCH(tag, LOC_EX_PART_TAG, "Local exchange particles");
      recvbuf[recvidx].n = nrecvpart[recvidx];
      dyndatasiz =
          dd_async_exchange_insert_particles(&recvbuf[recvidx], 0, source);
      if (dyndatasiz > 0) {
        recvbuf_dyn[recvidx].resize(dyndatasiz);
        MPI_Irecv(recvbuf_dyn[recvidx].data(), dyndatasiz, MPI_INT, source,
                  LOC_EX_DYN_TAG, comm_cart, &rreq[recvidx]);
        commstat.next(recvidx);
      }
      break;
    case CommunicationStatus::ReceiveStatus::RECV_DYNDATA:
      DIE_IF_TAG_MISMATCH(tag, LOC_EX_DYN_TAG, "Local exchange dyndata");
      dd_async_exchange_insert_dyndata(&recvbuf[recvidx], recvbuf_dyn[recvidx]);
      commstat.next(recvidx);
      break;
    default:
      std::cerr << "[" << this_node << "]"
                << "Unknown comm status for receive index " << recvidx
                << std::endl;
      break;
    }
  }
  MPI_Waitall(3 * num_comm_proc, sreq.data(), MPI_STATUS_IGNORE);

  // clear all buffers and free memory
  for (int i = 0; i < num_comm_proc; ++i) {
    // Remove particles from this nodes local list and free data
    for (int p = 0; p < sendbuf[i].n; p++) {
      local_particles[sendbuf[i].part[p].p.identity] = nullptr;
      free_particle(&sendbuf[i].part[p]);
    }
    realloc_particlelist(&sendbuf[i], 0);
    realloc_particlelist(&recvbuf[i], 0);
    sendbuf_dyn[i].clear();
    recvbuf_dyn[i].clear();
  }

#ifdef ADDITIONAL_CHECKS
  check_particle_consistency();
#endif
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_global_exchange_part(ParticleList *pl)
{
  // Prepare send/recv buffers to ALL processes
  std::vector<ParticleList> sendbuf(n_nodes), recvbuf(n_nodes);
  std::vector<std::vector<int>> sendbuf_dyn(n_nodes), recvbuf_dyn(n_nodes);
  std::vector<MPI_Request> sreq(3 * n_nodes, MPI_REQUEST_NULL);
  std::vector<MPI_Request> rreq(n_nodes, MPI_REQUEST_NULL);
  std::vector<int> nrecvpart(n_nodes, 0);

  for (int i = 0; i < n_nodes; ++i) {
    init_particlelist(&sendbuf[i]);
    init_particlelist(&recvbuf[i]);

    MPI_Irecv(&nrecvpart[i], 1, MPI_INT, i, GLO_EX_CNT_TAG, comm_cart,
              &rreq[i]);
  }

  // If pl == NULL do nothing, since there are no particles and cells on this
  // node
  if (pl) {
    // Find the correct node for each particle in pl
    for (int p = 0; p < pl->n; p++) {
      Particle *part = &pl->part[p];
      // fold_position(part->r.p, part->l.i);
      int rank = dd_p4est_pos_to_proc(part->r.p.data());
      if (rank != this_node) {  // It is actually a remote particle -> copy all
                                // data to sendbuffer
        if (rank > n_nodes || rank < 0) {
          fprintf(stderr, "process %i invalid\n", rank);
          errexit();
        }
        sendbuf_dyn[rank].insert(sendbuf_dyn[rank].end(), part->bl.e,
                                 part->bl.e + part->bl.n);
#ifdef EXCLUSIONS
        sendbuf_dyn[rank].insert(sendbuf_dyn[rank].end(), part->el.e,
                                 part->el.e + part->el.n);
#endif
        int pid = part->p.identity;
        move_indexed_particle(&sendbuf[rank], pl, p);
        local_particles[pid] = nullptr;
        if (p < pl->n)
          p -= 1;
      }
    }
  }

  // send number of particles, particles, and particle data
  for (int i = 0; i < n_nodes; ++i) {
    MPI_Isend(&sendbuf[i].n, 1, MPI_INT, i, GLO_EX_CNT_TAG, comm_cart, &sreq[i]);
    if (sendbuf[i].n <= 0)
      continue;
    MPI_Isend(sendbuf[i].part, sendbuf[i].n * sizeof(Particle), MPI_BYTE, i,
              GLO_EX_PART_TAG, comm_cart, &sreq[i + n_nodes]);
    if (sendbuf_dyn[i].size() <= 0)
      continue;
    MPI_Isend(sendbuf_dyn[i].data(), sendbuf_dyn[i].size(), MPI_INT, i,
              GLO_EX_DYN_TAG, comm_cart, &sreq[i + 2 * n_nodes]);
  }

  // Receive all data. The async communication scheme is the same as in
  // exchange_and_sort_particles
  MPI_Status status;
  CommunicationStatus commstat(n_nodes);
  while (true) {
    int recvidx;
    MPI_Waitany(n_nodes, rreq.data(), &recvidx, &status);
    if (recvidx == MPI_UNDEFINED)
      break;

    int dyndatasiz, source = status.MPI_SOURCE, tag = status.MPI_TAG;

    switch (commstat.expected(recvidx)) {
    case CommunicationStatus::ReceiveStatus::RECV_COUNT:
      DIE_IF_TAG_MISMATCH(tag, GLO_EX_CNT_TAG, "Global exchange count");
      if (nrecvpart[recvidx] > 0) {
        realloc_particlelist(&recvbuf[recvidx], nrecvpart[recvidx]);
        MPI_Irecv(recvbuf[recvidx].part, nrecvpart[recvidx] * sizeof(Particle),
                  MPI_BYTE, source, GLO_EX_PART_TAG, comm_cart, &rreq[recvidx]);
        commstat.next(recvidx);
      }
      break;
    case CommunicationStatus::ReceiveStatus::RECV_PARTICLES:
      DIE_IF_TAG_MISMATCH(tag, GLO_EX_PART_TAG, "Global exchange particles");
      recvbuf[recvidx].n = nrecvpart[recvidx];
      dyndatasiz =
          dd_async_exchange_insert_particles(&recvbuf[recvidx], 0, source);
      if (dyndatasiz > 0) {
        recvbuf_dyn[recvidx].resize(dyndatasiz);
        MPI_Irecv(recvbuf_dyn[recvidx].data(), dyndatasiz, MPI_INT, source,
                  GLO_EX_DYN_TAG, comm_cart, &rreq[recvidx]);
        commstat.next(recvidx);
      }
      break;
    case CommunicationStatus::ReceiveStatus::RECV_DYNDATA:
      DIE_IF_TAG_MISMATCH(tag, GLO_EX_DYN_TAG, "Global exchange dyndata");
      dd_async_exchange_insert_dyndata(&recvbuf[recvidx], recvbuf_dyn[recvidx]);
      commstat.next(recvidx);
      break;
    default:
      std::cerr << "[" << this_node << "]"
                << "Unknown comm status for receive index " << recvidx
                << std::endl;
      break;
    }
  }

  MPI_Waitall(3 * n_nodes, sreq.data(), MPI_STATUS_IGNORE);

  for (int i = 0; i < n_nodes; ++i) {
    // Remove particles from this nodes local list and free data
    for (int p = 0; p < sendbuf[i].n; p++) {
      local_particles[sendbuf[i].part[p].p.identity] = nullptr;
      free_particle(&sendbuf[i].part[p]);
    }
    realloc_particlelist(&sendbuf[i], 0);
    realloc_particlelist(&recvbuf[i], 0);
  }
}

//--------------------------------------------------------------------------------------------------
void dd_p4est_exchange_and_sort_particles (int global_flag) {
  if (global_flag == CELL_GLOBAL_EXCHANGE) { // perform a global resorting
    ParticleList sendbuf;
    init_particlelist(&sendbuf);
    // Go through all local particles and move those not on node to the sendbuf
    for (int c = 0; c < local_cells.n; c++) {
      Cell *pl = local_cells.cell[c];
      for (int p = 0; p < pl->n; p++) {
        //fold_position(pl->part[p].r.p, pl->part[p].l.i);
        Cell *nc = dd_p4est_save_position_to_cell(pl->part[p].r.p.data());
        if (nc == nullptr) { // Belongs to other node
          int pid = pl->part[p].p.identity;
          move_indexed_particle(&sendbuf, pl, p);
          local_particles[pid] = nullptr;
          if(p < pl->n) p -= 1;
        } else if(nc != pl) { // Still on node but particle belongs to other cell
          move_indexed_particle(nc, pl, p);
          if(p < pl->n) p -= 1;
        }
        // Else not required since particle in cell it belongs to
      }
    }
    // Invoke communication, the particle index is cleared there
    if (local_cells.n > 0)
      dd_p4est_global_exchange_part(&sendbuf);
    else // Still call it if there are no cells on this node
      dd_p4est_global_exchange_part(nullptr);
    // Free remaining particles in sendbuf, but it should be empty already
    for (int p = 0; p < sendbuf.n; p++) {
      local_particles[sendbuf.part[p].p.identity] = nullptr;
      free_particle(&sendbuf.part[p]);
    }
    realloc_particlelist(&sendbuf, 0);
  } else { // do the local resorting (particles move at most on cell)
    dd_p4est_local_exchange_and_sort_particles();
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_repart_exchange_part (CellPList *old) {
  std::vector<int> send_quads(n_nodes); //< send_quads[i]: Number of cells sent to process i
  std::vector<int> send_prefix(n_nodes + 1);
  std::vector<int> recv_quads(n_nodes); //< recv_quads[i]: Number of cells received from process i
  // Prefix of cells. Cells to be received are possibly at the front and at the back of the 
  // contiguous inverval along the SFC. This array counts the cell number of the overlap of
  // Different domains.
  std::vector<int> recv_prefix(n_nodes + 1);

  // Map: (i, c) -> Number of particles for process i in cell c
  std::vector<std::vector<int>> send_num_part(n_nodes);
  // Map: (i, c) -> Number of particles from process i in cell c
  std::vector<std::vector<int>> recv_num_part(n_nodes);

  int lb_old_local = old_global_first_quadrant[this_node];
  int ub_old_local = old_global_first_quadrant[this_node + 1];
  int lb_new_local = ds.p4est->global_first_quadrant[this_node];
  int ub_new_local = ds.p4est->global_first_quadrant[this_node + 1];
  int lb_old_remote = 0;
  int ub_old_remote = 0;
  int lb_new_remote = 0;
  int ub_new_remote = 0;

  std::vector<MPI_Request> sreq(3 * n_nodes, MPI_REQUEST_NULL);
  std::vector<MPI_Request> rreq(n_nodes, MPI_REQUEST_NULL);

  std::vector<ParticleList> sendbuf(n_nodes), recvbuf(n_nodes);
  std::vector<std::vector<int>> sendbuf_dyn(n_nodes), recvbuf_dyn(n_nodes);

  // determine from which processors we receive quadrants
  /** there are 5 cases to distinguish
   * 1. no quadrants of neighbor need to be received; neighbor rank < rank
   * 2. some quadrants of neighbor need to be received; neighbor rank < rank
   * 3. all quadrants of neighbor need to be received from neighbor
   * 4. some quadrants of neighbor need to be received; neighbor rank > rank
   * 5. no quadrants of neighbor need to be received; neighbor rank > rank
   */
  recv_prefix[0] = 0;
  for (int p = 0; p < n_nodes; ++p) {
    lb_old_remote = ub_old_remote;
    ub_old_remote = old_global_first_quadrant[p + 1];

    recv_quads[p] = std::max(0,
                           std::min(ub_old_remote, ub_new_local) -
                               std::max(lb_old_remote, lb_new_local));
    recv_num_part[p].resize(recv_quads[p]);
    init_particlelist(&recvbuf[p]);
    recv_prefix[p+1] = recv_prefix[p] + recv_quads[p];
    if (p != this_node && recv_quads[p] > 0) {
      MPI_Irecv(recv_num_part[p].data(),
                recv_quads[p], MPI_INT, p, REP_EX_CNT_TAG,
                comm_cart, &rreq[p]);
      //fprintf(stderr, "[%i] : recv %i (%i)\n", this_node, p, REP_EX_CNT_TAG);
    }
  }

  // send respective quadrants to other processors
  send_prefix[0] = 0;
  int c_cnt = 0;
  for (int p = 0; p < n_nodes; ++p) {
    lb_new_remote = ub_new_remote;
    ub_new_remote = ds.p4est->global_first_quadrant[p + 1];

    send_quads[p] = std::max(0,
                           std::min(ub_old_local, ub_new_remote) -
                               std::max(lb_old_local, lb_new_remote));
    send_prefix[p+1] = send_prefix[p] + send_quads[p];

    // Fill send list for number of particles per cell
    send_num_part[p].resize(send_quads[p]);
    init_particlelist(&sendbuf[p]);
    int send_sum = 0, send_inc = 0;
    for (int c = 0; c < send_quads[p]; ++c) {
      if (p == this_node) {
        recv_num_part[p][c] = old->cell[c_cnt]->n;
        //realloc_particlelist(&cells[recv_prefix[p] + c], old->cell[c_cnt]->n);
        //cells[recv_prefix[p] + c].n = old->cell[c_cnt]->n;
      } else {
        //realloc_particlelist(&sendbuf[p], sendbuf[p].n + old->cell[c_cnt]->n);
        //sendbuf[p].n = old->cell[c_cnt]->n;
      }
      send_num_part[p][c] = old->cell[c_cnt]->n;
      send_sum += send_num_part[p][c];
      for (int i = 0; i < old->cell[c_cnt]->n; i++) {
        Particle *part = &old->cell[c_cnt]->part[i];
        if (p != this_node) {
          send_inc+=1;
          // It is actually a remote particle -> copy all data to sendbuffer
          sendbuf_dyn[p].insert(sendbuf_dyn[p].end(), part->bl.e,
                                part->bl.e + part->bl.n);
#ifdef EXCLUSIONS
          sendbuf_dyn[p].insert(sendbuf_dyn[p].end(), part->el.e,
                                part->el.e + part->el.n);
#endif
          int pid = part->p.identity;
          //memcpy(&sendbuf[p].part[i], part, sizeof(Particle));
          append_unindexed_particle(&sendbuf[p], std::move(*part));
          local_particles[pid] = nullptr;
        } else { // Particle stays local
          //int pid = part->p.identity;
          //memcpy(&cells[recv_prefix[p] + c].part[i], part, sizeof(Particle));
          //local_particles[pid] = &cells[recv_prefix[p] + c].part[i];
          //cells[recv_prefix[p] + c].n += 1;
          append_indexed_particle(&cells[recv_prefix[p] + c], std::move(*part));
        }
      }
      ++c_cnt;
    }
    if (p != this_node && send_sum != sendbuf[p].n) {
      fprintf(stderr, "[%i] send buffer (%i) mismatch for process %i (sum %i, inc %i)\n", this_node, p, sendbuf[p].n, send_sum, send_inc);
      errexit();
    }
    if (p != this_node && send_quads[p] > 0) {
      MPI_Isend(send_num_part[p].data(),
                send_quads[p], MPI_INT, p, REP_EX_CNT_TAG,
                comm_cart, &sreq[p]);
      if (sendbuf[p].n > 0) {
        MPI_Isend(sendbuf[p].part, 
                  sendbuf[p].n * sizeof(Particle), MPI_BYTE, p, REP_EX_PART_TAG,
                  comm_cart, &sreq[p + n_nodes]);
        if (sendbuf_dyn[p].size() > 0) {
          MPI_Isend(sendbuf_dyn[p].data(), 
                    sendbuf_dyn[p].size(), MPI_INT, p, REP_EX_DYN_TAG,
                    comm_cart, &sreq[p + 2 * n_nodes]);
        }
      }
    }
  }

  // Receive all data. The async communication scheme is the same as in
  // exchange_and_sort_particles
  MPI_Status status;
  int read, dyndatasiz;
  CommunicationStatus commstat(n_nodes);
  while (true) {
    int recvidx;
    MPI_Waitany(n_nodes, rreq.data(), &recvidx, &status);
    if (recvidx == MPI_UNDEFINED)
      break;

    int dyndatasiz, source = status.MPI_SOURCE, tag = status.MPI_TAG;

    switch (commstat.expected(recvidx)) {
    case CommunicationStatus::ReceiveStatus::RECV_COUNT:
      DIE_IF_TAG_MISMATCH(tag, REP_EX_CNT_TAG, "Repart exchange count");
      if (recv_quads[source] > 0) {
        int sum = std::accumulate(recv_num_part[source].begin(), recv_num_part[source].end(), 0);
        recvbuf[source].n = sum;
        realloc_particlelist(&recvbuf[source], sum);
        if (sum > 0) {
          MPI_Irecv(recvbuf[source].part, sum * sizeof(Particle),
                    MPI_BYTE, source, REP_EX_PART_TAG, comm_cart, &rreq[recvidx]);
          commstat.next(recvidx);
        }
      }
      break;
    case CommunicationStatus::ReceiveStatus::RECV_PARTICLES:
      DIE_IF_TAG_MISMATCH(tag, REP_EX_PART_TAG, "Repart exchange particles");
      dyndatasiz = 0;
      read = 0;
      for (int c = 0; c < recv_quads[source]; ++c) {
        for (int p = 0; p < recv_num_part[source][c]; ++p) {
          dyndatasiz += recvbuf[source].part[read + p].bl.n;
#ifdef EXCLUSIONS
          dyndatasiz += recvbuf[source].part[read + p].el.n;
#endif
          append_indexed_particle(&cells[recv_prefix[source] + c], std::move(recvbuf[source].part[read + p]));
        }
        read += recv_num_part[source][c];
      }
      if (dyndatasiz > 0) {
        recvbuf_dyn[source].resize(dyndatasiz);
        MPI_Irecv(recvbuf_dyn[source].data(), dyndatasiz, MPI_INT, source,
                  REP_EX_DYN_TAG, comm_cart, &rreq[recvidx]);
        commstat.next(recvidx);
      }
      break;
    case CommunicationStatus::ReceiveStatus::RECV_DYNDATA:
      DIE_IF_TAG_MISMATCH(tag, REP_EX_DYN_TAG, "Repart exchange dyndata");
      read = 0;
      for (int c = 0; c < recv_quads[source]; ++c) {
        for (int i = 0; i < recv_num_part[source][c]; ++i) {
          Particle *p = &cells[recv_prefix[source] + c].part[i];
          if (p->bl.n > 0) {
            if (!(p->bl.e = (int *) malloc(p->bl.max * sizeof(int)))) {
              fprintf(stderr, "Tod.\n");
              errexit();
            }
            memcpy(p->bl.e, &recvbuf_dyn[source][read], p->bl.n * sizeof(int));
            read += p->bl.n;
          } else {
            p->bl.e = nullptr;
            p->bl.max = 0;
          }
#ifdef EXCLUSIONS
          if (p->el.n > 0) {
            if (!(p->el.e = (int *) malloc(p->el.max * sizeof(int)))) {
              fprintf(stderr, "Tod.\n");
              errexit();
            }
            memcpy(p->el.e, &recvbuf_dyn[source][read], p->el.n*sizeof(int));
            read += p->el.n;
          } else {
            p->el.e = nullptr;
            p->el.max = 0;
          }
#endif
        }
      }
      commstat.next(recvidx);
      break;
    default:
      std::cerr << "[" << this_node << "]"
                << "Unknown comm status for receive index " << recvidx
                << std::endl;
      break;
    }
  }
  MPI_Waitall(3 * n_nodes, sreq.data(), MPI_STATUS_IGNORE);

  for (int i = 0; i < n_nodes; ++i) {
    // Remove particles from this nodes local list and free data
    for (int p = 0; p < sendbuf[i].n; p++) {
      local_particles[sendbuf[i].part[p].p.identity] = nullptr;
      free_particle(&sendbuf[i].part[p]);
    }
    realloc_particlelist(&sendbuf[i], 0);
    realloc_particlelist(&recvbuf[i], 0);
  }

#ifdef ADDITIONAL_CHECKS
  check_particle_consistency();
#endif
}
//--------------------------------------------------------------------------------------------------
// Maps a position to the Cartesian grid and returns the Morton index of this
// coordinates.
// Note: The global Morton index returned here is NOT equal to the local cell
//       index!!!
static uint64_t dd_p4est_pos_morton_idx(double pos[3]) {
  double pfold[3] = {pos[0], pos[1], pos[2]};
  int im[3] = {0, 0, 0}; /* dummy */
  fold_position(pfold, im);
  for (int d = 0; d < 3; ++d) {
    double errmar = 0.5 * ROUND_ERROR_PREC * box_l[d];
    if (pos[d] < 0 && pos[d] > -errmar)
      pfold[d] = 0;
    else if (pos[d] >= box_l[d] && pos[d] < box_l[d] + errmar)
      pfold[d] = pos[d] - 0.5 * dd.cell_size[d];
    // In the other two cases ("pos[d] <= -errmar" and
    // "pos[d] >= box_l[d] + errmar") pfold is correct.
  }
  return dd_p4est_cell_morton_idx(pfold[0] * dd.inv_cell_size[0],
                                  pfold[1] * dd.inv_cell_size[1],
                                  pfold[2] * dd.inv_cell_size[2]);
}
//--------------------------------------------------------------------------------------------------
// Find the process that handles the position
int dd_p4est_pos_to_proc(double pos[3]) {
  auto it = std::upper_bound(
      std::begin(first_morton_idx_of_rank), std::end(first_morton_idx_of_rank) - 1,
      dd_p4est_pos_morton_idx(pos), [](uint64_t i, uint64_t idx) { return i < idx; });

  return std::distance(std::begin(first_morton_idx_of_rank), it) - 1;
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_topology_init(CellPList *old, bool isRepart) {

  int c,p,np;
  int exchange_data, update_data;
  Particle *part;

  // use p4est_dd callbacks, but Espresso sees a DOMDEC
  cell_structure.type             = CELL_STRUCTURE_P4EST;
  cell_structure.position_to_node = dd_p4est_pos_to_proc;
  cell_structure.position_to_cell = dd_p4est_position_to_cell;
  //cell_structure.position_to_cell = dd_p4est_save_position_to_cell;

  /* set up new domain decomposition cell structure */
  dd_p4est_create_grid(isRepart);
  /* mark cells */
  dd_p4est_mark_cells();

  /* create communicators */
  dd_p4est_prepare_comm(&cell_structure.ghost_cells_comm, GHOSTTRANS_PARTNUM);

  exchange_data = (GHOSTTRANS_PROPRTS | GHOSTTRANS_POSITION | GHOSTTRANS_POSSHFTD);
  update_data   = (GHOSTTRANS_POSITION | GHOSTTRANS_POSSHFTD);

  dd_p4est_prepare_comm(&cell_structure.exchange_ghosts_comm,  exchange_data);
  dd_p4est_prepare_comm(&cell_structure.update_ghost_pos_comm, update_data);
  dd_p4est_prepare_comm(&cell_structure.collect_ghost_force_comm, GHOSTTRANS_FORCE);

  /* collect forces has to be done in reverted order! */
  dd_p4est_revert_comm_order(&cell_structure.collect_ghost_force_comm);
#ifdef LB
  dd_p4est_prepare_comm(&cell_structure.ghost_lbcoupling_comm, GHOSTTRANS_COUPLING);
#endif

#ifdef IMMERSED_BOUNDARY
  // Immersed boundary needs to communicate the forces from but also to the ghosts
  // This is different than usual collect_ghost_force_comm (not in reverse order)
  // Therefore we need our own communicator
  dd_p4est_prepare_comm(&cell_structure.ibm_ghost_force_comm, GHOSTTRANS_FORCE);
#endif

#ifdef ENGINE
  dd_p4est_prepare_comm(&cell_structure.ghost_swimming_comm, GHOSTTRANS_SWIMMING);
#endif

  /* initialize cell neighbor structures */
  dd_p4est_init_cell_interactions();

  if (isRepart) {
    dd_p4est_repart_exchange_part(old);
    for(c=0; c<local_cells.n; c++)
      update_local_particles(local_cells.cell[c]);
  } else {
    // Go through all old particles and find owner & cell
    for (c = 0; c < old->n; c++) {
      part = old->cell[c]->part;
      np   = old->cell[c]->n;
      for (p = 0; p < np; p++) {
        fold_position(part[p].r.p, part[p].l.i);

        Cell *nc = dd_p4est_save_position_to_cell(part[p].r.p.data());
        if (nc == nullptr) { // Particle is on other process, move it to cell[0]
          append_unindexed_particle(local_cells.cell[0], std::move(part[p]));
        } else { // It is on this node, move it to right local_cell
          append_unindexed_particle(nc, std::move(part[p]));
        }
      }
    }
    // Create particle index
    for(c=0; c<local_cells.n; c++) {
      update_local_particles(local_cells.cell[c]);
    }
    // Invoke global communication for cell[0] containing particles of other processes
    if (local_cells.n > 0)
      dd_p4est_global_exchange_part(local_cells.cell[0]);
    else // Call it anyway in case there are no cells on this node (happens during startup)
      dd_p4est_global_exchange_part(nullptr);
  }
}
#endif

//--------------------------------------------------------------------------------------------------
void dd_p4est_write_parallel_vtk(char *filename) {
#ifdef DD_P4EST
  /* strip file ending from filename (if given) */
  char *pos_file_ending;
  pos_file_ending = strrchr(filename, '.');
  if (pos_file_ending != nullptr) {
    *pos_file_ending = '\0';
  }
  int len = static_cast<int>(strlen(filename));
  ++len;

  /* call mpi printing routine on all slaves and communicate the filename */
  mpi_call(mpi_dd_p4est_write_particle_vtk, -1, len);

  MPI_Bcast(filename, len, MPI_CHAR, 0, comm_cart);
  dd_p4est_write_particle_vtk(filename);
#endif
}

#ifdef DD_P4EST
void dd_p4est_write_particle_vtk(char *filename) {
  char fname[1024];
  // node 0 writes the header file
  if (!this_node) {
    sprintf(fname, "%s.pvtp", filename);
    FILE *h = fopen(fname, "w");
    fprintf(h, "<?xml version=\"1.0\"?>\n");
    fprintf(h, "<VTKFile type=\"PPolyData\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n\t");
    fprintf(h, "<PPolyData GhostLevel=\"0\">\n\t\t<PPoints>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Float32\" Name=\"Position\" "
               "NumberOfComponents=\"3\" format=\"ascii\"/>\n");
    fprintf(h, "\t\t</PPoints>\n\t\t");
    fprintf(h, "<PPointData Scalars=\"mpirank,part_id,cell_id\" "
               "Vectors=\"velocity\">\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Int32\" Name=\"mpirank\" "
               "format=\"ascii\"/>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Int32\" Name=\"part_id\" "
               "format=\"ascii\"/>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Int32\" Name=\"cell_id\" "
               "format=\"ascii\"/>\n\t\t\t");
    fprintf(h, "<PDataArray type=\"Float32\" Name=\"velocity\" "
               "NumberOfComponents=\"3\" format=\"ascii\"/>\n\t\t");
    fprintf(h, "</PPointData>\n");
    for (int p = 0; p < n_nodes; ++p)
      fprintf(h, "\t\t<Piece Source=\"%s_%04i.vtp\"/>\n", filename, p);
    fprintf(h, "\t</PPolyData>\n</VTKFile>\n");
    fclose(h);
  }
  // write the actual parallel particle files
  sprintf(fname, "%s_%04i.vtp", filename, this_node);
  int num_p = 0;
  for (int c = 0; c < local_cells.n; c++) {
    num_p += local_cells.cell[c]->n;
  }
  FILE *h = fopen(fname, "w");
  fprintf(h, "<?xml version=\"1.0\"?>\n");
  fprintf(h, "<VTKFile type=\"PolyData\" version=\"0.1\" "
             "byte_order=\"LittleEndian\">\n\t");
  fprintf(h,
          "<PolyData>\n\t\t<Piece NumberOfPoints=\"%i\" NumberOfVerts=\"0\" ",
          num_p);
  fprintf(h, "NumberOfLines=\"0\" NumberOfStrips=\"0\" "
             "NumberOfPolys=\"0\">\n\t\t\t<Points>\n\t\t\t\t");
  fprintf(h, "<DataArray type=\"Float32\" Name=\"Position\" "
             "NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p = 0; p < np; ++p) {
      std::array <double, 3> t_pos = boxl_to_treecoords_copy(part[p].r.p.data());
      fprintf(h, "\t\t\t\t\t%le %le %le\n", t_pos[0], t_pos[1], t_pos[2]);
    }
  }
  fprintf(h, "\t\t\t\t</DataArray>\n\t\t\t</Points>\n\t\t\t");
  fprintf(h, "<PointData Scalars=\"mpirank,part_id,cell_id\" "
             "Vectors=\"velocity\">\n\t\t\t\t");
  fprintf(h, "<DataArray type=\"Int32\" Name=\"mpirank\" "
             "format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "%i ", this_node);
    }
  }
  fprintf(h, "\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Int32\" "
             "Name=\"part_id\" format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "%i ", part[p].p.identity);
    }
  }
  fprintf(h, "\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Int32\" "
             "Name=\"cell_id\" format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "%i ", c);
    }
  }
  fprintf(h, "\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Float32\" "
             "Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p = 0; p < np; ++p) {
      fprintf(h, "\t\t\t\t\t%le %le %le\n", part[p].m.v[0] / time_step,
              part[p].m.v[1] / time_step, part[p].m.v[2] / time_step);
    }
  }
  fprintf(h, "\t\t\t\t</DataArray>\n\t\t\t</PointData>\n");
  fprintf(h, "\t\t</Piece>\n\t</PolyData>\n</VTKFile>\n");
  fclose(h);
}

//--------------------------------------------------------------------------------------------------
// This is basically a copy of the dd_on_geometry_change
void dd_p4est_on_geometry_change(int flags) {
#ifndef LB_ADAPTIVE
  // Reinit only if max_range or box_l changes such that the old cell system
  // is not valid anymore.
  if (flags & CELL_FLAG_GRIDCHANGED
      || grid_size[0] != std::max<int>(box_l[0] / max_range, 1)
      || grid_size[1] != std::max<int>(box_l[1] / max_range, 1)
      || grid_size[2] != std::max<int>(box_l[2] / max_range, 1)) {
    cells_re_init(CELL_STRUCTURE_CURRENT);
  }
#else
  cells_re_init(CELL_STRUCTURE_CURRENT);
#endif
}

void
p4est_dd_repart_preprocessing()
{
  // Save global_first_quadrants for migration
  old_global_first_quadrant.clear();
  std::copy_n(ds.p4est->global_first_quadrant, n_nodes + 1,
            std::back_inserter(old_global_first_quadrant));
}

//--------------------------------------------------------------------------------------------------
// Purely MD Repartitioning functions follow now.
//-------------------------------------------------------------------------------------------------
// Metric has to hold num_local_cells many elements.
// Fills the global std::vector repart_nquads to be used by the next
// cellsystem re-init in conjunction with p4est_partition_given.
static void
p4est_dd_repart_calc_nquads(const std::vector<double>& metric, bool debug)
{
  if (metric.size() != num_local_cells) {
    std::cerr << "Error in provided metric: too few elements." << std::endl;
    part_nquads.clear();
    return;
  }

  // Determine prefix and target load
  double localsum = std::accumulate(metric.begin(), metric.end(), 0.0);
  double sum, prefix = 0; // Initialization is necessary on rank 0!
  MPI_Allreduce(&localsum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
  MPI_Exscan(&localsum, &prefix, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
  double target = sum / n_nodes;

  if (target == 0.0) {
    if (this_node == 0)
      std::cerr << "p4est_dd_repart: Metric all-zero. Aborting repart." << std::endl;
    // Leave part_nquads as it is and exit.
    return;
  }

  if (debug) {
    printf("[%i] NCells: %zu\n", this_node, num_local_cells);
    printf("[%i] Local : %lf\n", this_node, localsum);
    printf("[%i] Global: %lf\n", this_node, sum);
    printf("[%i] Target: %lf\n", this_node, target);
    printf("[%i] Prefix: %lf\n", this_node, prefix);
  }

  part_nquads.resize(n_nodes);
  std::fill(part_nquads.begin(), part_nquads.end(),
            static_cast<p4est_locidx_t>(0));

  // Determine new process boundaries in local subdomain
  // Evaluated for its side effect of setting part_nquads.
  std::accumulate(metric.begin(), metric.end(),
                  prefix,
                  [target](double cellpref, double met_i) {
                    int proc = std::min<int>(cellpref / target, n_nodes - 1);
                    part_nquads[proc]++;
                    return cellpref + met_i;
                  });

  MPI_Allreduce(MPI_IN_PLACE, part_nquads.data(), n_nodes,
                P4EST_MPI_LOCIDX, MPI_SUM, comm_cart);

  // Could try to steal quads from neighbors.
  // Global reshifting (i.e. stealing from someone else than the direct
  // neighbors) is not a good idea since it globally changes the metric.
  // Anyways, this is most likely due to a bad quad/proc quotient.
  if (part_nquads[this_node] == 0) {
    fprintf(stderr, "[%i] No quads assigned to me. Cannot guarantee to work. Exiting\n", this_node);
    fprintf(stderr, "[%i] Try changing the metric or reducing the number of processes\n", this_node);
    errexit();
  }

  if (debug) {
    printf("[%i] Nquads: %i\n", this_node, part_nquads[this_node]);

    if (this_node == 0) {
      p4est_gloidx_t totnquads = std::accumulate(part_nquads.begin(),
                                                 part_nquads.end(),
                                                 static_cast<p4est_gloidx_t>(0));
      if (ds.p4est->global_num_quadrants != totnquads) {
        fprintf(stderr,
                "[%i] ERROR: totnquads = %li but global_num_quadrants = %li\n",
                this_node, totnquads, ds.p4est->global_num_quadrants);
        errexit();
      }
    }
  }
}
//--------------------------------------------------------------------------------------------------
// Repartition the MD grd:
// Evaluate metric and set part_nquads global vector to be used by a
// reinit of the cellsystem. Afterwards, directly reinits the cellsystem.
void
p4est_dd_repartition(const std::string& desc, bool verbose)
{
  std::vector<double> weights = repart::metric{desc}();
  p4est_dd_repart_calc_nquads(weights, false && verbose);
  
  if (verbose && this_node == 0) {
    std::cout << " New ncells per proc: ";
    std::copy(std::begin(part_nquads), std::end(part_nquads), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }

  cells_re_init(CELL_STRUCTURE_CURRENT, true);
}

#endif

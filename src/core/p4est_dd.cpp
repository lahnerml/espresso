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
#include <p8est_vtk.h>
#include <p8est_extended.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#include "call_trace.hpp"
#include "domain_decomposition.hpp"
#include "ghosts.hpp"
#include "repart.hpp"
//--------------------------------------------------------------------------------------------------
#define CELLS_MAX_NEIGHBORS 14
//--------------------------------------------------------------------------------------------------
static int brick_size[3]; // number of trees in each direction
static int grid_size[3];  // number of quadrants/cells in each direction
static std::vector<int> p4est_space_idx; // n_nodes + 1 elements, storing the first Morton index of local cell
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
// Internal p4est structure
typedef struct {
  int rank; // the node id
  int lidx; // local index
  int ishell[26]; // local or ghost index of fullshell in p4est
  int rshell[26]; // rank of fullshell cells
} quad_data_t;
//--------------------------------------------------------------------------------------------------
// Datastructure for partitioning holding the number of quads per process
// Empty as long as tclcommand_repart has not been called.
static std::vector<p4est_locidx_t> part_nquads;
//--------------------------------------------------------------------------------------------------
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
int dd_p4est_full_shell_neigh(int cell, int neighidx)
{
    if (neighidx >= 0 && neighidx < 26)
        return dd.p4est_shell[cell].neighbor[neighidx];
    else if (neighidx == 26)
        return cell;
    else {
        fprintf(stderr, "dd_p4est_full_shell_neigh: Require 0 <= neighidx < 27.\n");
        errexit();
    }
}
//-----------------------------------------------------------------------------------------
// Init Callback for internal structure
static void init_fn (p4est_t* p4est, p4est_topidx_t tree, p4est_quadrant_t* q) {
  ((quad_data_t*)(q->p.user_data))->rank = this_node;
  ((quad_data_t*)(q->p.user_data))->lidx = -1;
  for (int i=0;i<26;++i) {
    ((quad_data_t*)(q->p.user_data))->ishell[i] = -1;
    ((quad_data_t*)(q->p.user_data))->rshell[i] = -1;
  }
}
//--------------------------------------------------------------------------------------------------
static inline int count_trailing_zeros(int x)
{
  int z = 0;
  for (; (x & 1) == 0; x >>= 1) z++;
  return z;
}

// Compute the grid- and bricksize according to box_l and maxrange
int dd_p4est_cellsize_optimal() {
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
  int lvl = count_trailing_zeros(ncells[0] | ncells[1] | ncells[2]);

  brick_size[0] = ncells[0] >> lvl;
  brick_size[1] = ncells[1] >> lvl;
  brick_size[2] = ncells[2] >> lvl;

  return lvl; // return the level of the grid
}
//--------------------------------------------------------------------------------------------------
// Creates a forest with box_l trees in each direction
int dd_p4est_cellsize_even () {
  brick_size[0] = box_l[0];
  brick_size[1] = box_l[1];
  brick_size[2] = box_l[2];

  int ncells = 1;
  if (max_range > ROUND_ERROR_PREC * box_l[0])
    ncells = std::max<int>(1.0 / max_range, 1);

  int lvl = Utils::nat_log2_floor(ncells);

  grid_size[0] = brick_size[0] << lvl;
  grid_size[1] = brick_size[1] << lvl;
  grid_size[2] = brick_size[2] << lvl;

  return lvl; // Return level > 0 if max_range <= 0.5
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_create_grid (bool isRepart) {
  //printf("%i : new MD grid\n", this_node);
  CALL_TRACE();
  // Clear data to prevent accidental use of old stuff
  comm_rank.clear();
  comm_proc.clear();
  comm_recv.clear();
  comm_send.clear();
  p4est_space_idx.clear();
  dd.p4est_shell.clear();

#ifdef LB_ADAPTIVE
  // the adaptive LB has a strange grid, thus we have to do something similar here
  if (max_range < 1.0)
    grid_level = dd_p4est_cellsize_even();
  else
    grid_level = dd_p4est_cellsize_optimal();
#else
  grid_level = dd_p4est_cellsize_optimal();
#endif
  
#ifndef P4EST_NOCHANGE
  // set global variables
  dd.cell_size[0] = box_l[0]/(double)grid_size[0];
  dd.cell_size[1] = box_l[1]/(double)grid_size[1];
  dd.cell_size[2] = box_l[2]/(double)grid_size[2];
  dd.inv_cell_size[0] = 1.0/dd.cell_size[0];
  dd.inv_cell_size[1] = 1.0/dd.cell_size[1];
  dd.inv_cell_size[2] = 1.0/dd.cell_size[2];
  max_skin = std::min(std::min(dd.cell_size[0],dd.cell_size[1]),dd.cell_size[2]) - max_cut;
  
  CELL_TRACE(printf("%i : gridsize %ix%ix%i\n", this_node, grid_size[0], grid_size[1], grid_size[2]));
  CELL_TRACE(printf("%i : bricksize %ix%ix%i level %i\n", this_node, brick_size[0], brick_size[1], brick_size[2], grid_level));
  CELL_TRACE(printf("%i : cellsize %lfx%lfx%lf\n", this_node, dd.cell_size[0], dd.cell_size[1], dd.cell_size[2]));
#endif
  
  // create p4est structs
  if (!isRepart) {
    auto oldconn = std::move(dd.p4est_conn);
    dd.p4est_conn =
        std::unique_ptr<p4est_connectivity_t>(p8est_connectivity_new_brick(
            brick_size[0], brick_size[1], brick_size[2], PERIODIC(0),
            PERIODIC(1), PERIODIC(2)));
    dd.p4est = std::unique_ptr<p4est_t>(
        p4est_new_ext(comm_cart, dd.p4est_conn, 0, grid_level, true,
                      sizeof(quad_data_t), init_fn, NULL));
  } else {
    // Save global_first_quadrants for migration
    old_global_first_quadrant.clear();
    std::copy_n(dd.p4est->global_first_quadrant, n_nodes + 1,
              std::back_inserter(old_global_first_quadrant));
  }
  // Repartition uniformly if part_nquads is empty (because not repart has been
  // done yet). Else use part_nquads as given partitioning.
  if (part_nquads.size() == 0)
    p4est_partition(dd.p4est, 0, NULL);
  else
    p4est_partition_given(dd.p4est, part_nquads.data());

  auto p4est_ghost = castable_unique_ptr<p4est_ghost_t>(
      p4est_ghost_new(dd.p4est, P8EST_CONNECT_CORNER));
  auto p4est_mesh = castable_unique_ptr<p4est_mesh_t>(
      p4est_mesh_new_ext(dd.p4est, p4est_ghost, 1, 1, 0, P8EST_CONNECT_CORNER));

  CELL_TRACE(printf("%i : %i %i-%i %i\n",
    this_node,periodic,p4est->first_local_tree,p4est->last_local_tree,p4est->local_num_quadrants));
  
  // create space filling inforamtion about first quads per node from p4est
  p4est_space_idx.resize(n_nodes + 1);
  for (int i=0;i<=n_nodes;++i) {
    p4est_quadrant_t *q = &dd.p4est->global_first_position[i];
    //p4est_quadrant_t c;
    if (i < n_nodes) {
      double xyz[3];
      p4est_qcoord_to_vertex(dd.p4est_conn,q->p.which_tree,q->x,q->y,q->z,xyz);
      //c.x = xyz[0]*(1<<grid_level);
      //c.y = xyz[1]*(1<<grid_level);
      //c.z = xyz[2]*(1<<grid_level);
      p4est_space_idx[i] = dd_p4est_cell_morton_idx(xyz[0]*(1<<grid_level),
                                                    xyz[1]*(1<<grid_level),xyz[2]*(1<<grid_level));
    } else {
      /*c.x = 1<<grid_level;
      while (c.x < grid_size[0]) c.x <<= 1;
      c.y = 0;
      c.z = 0;*/
      int64_t tmp = 1<<grid_level;
      while(tmp < grid_size[0]) tmp <<= 1;
      while(tmp < grid_size[1]) tmp <<= 1;
      while(tmp < grid_size[2]) tmp <<= 1;
      p4est_space_idx[i] = tmp*tmp*tmp;
    }
    //c.level = P4EST_QMAXLEVEL;
    //p4est_space_idx[i] = p4est_quadrant_linear_id(&c,P4EST_QMAXLEVEL+1);
    if (i == this_node + 1) {
      CELL_TRACE(printf("%i : %i - %i\n", this_node, p4est_space_idx[i-1], p4est_space_idx[i] - 1));
    }
  }
  
  // geather cell neighbors
  std::vector<uint64_t> quads;
  std::vector<local_shell_t> shell;
  
  // Loop all local cells to geather information for those
  for (int i=0;i<dd.p4est->local_num_quadrants;++i) {
    p4est_quadrant_t *q = p4est_mesh_get_quadrant(dd.p4est,p4est_mesh,i);
    quad_data_t *data = (quad_data_t*)(q->p.user_data);
    double xyz[3];
    p4est_qcoord_to_vertex(dd.p4est_conn, p4est_mesh->quad_to_tree[i], q->x, q->y, q->z, xyz);
    uint64_t ql = 1<<p4est_tree_array_index(dd.p4est->trees,p4est_mesh->quad_to_tree[i])->maxlevel;
    uint64_t x = xyz[0]*ql;
    uint64_t y = xyz[1]*ql;
    uint64_t z = xyz[2]*ql;
    // This is a simple but easy unique index, that also works for cells outside box_l (by one cell)
    quads.push_back((x+1) | ((y+1)<<21) | ((z+1)<<42));
    local_shell_t ls;
    ls.idx = i;
    ls.rank = this_node;
    ls.shell = 0;
    ls.boundary = 0;
    ls.coord[0] = x;
    ls.coord[1] = y;
    ls.coord[2] = z;
    ls.p_cnt = 0;
    // Geather all inforamtion about neighboring cells
    for (int n=0;n<26;++n) {
      ls.neighbor[n] = -1;
      sc_array_t *ne, *ni;
      ne = sc_array_new(sizeof(int));
      ni = sc_array_new(sizeof(int));
      p4est_mesh_get_neighbors(dd.p4est, p4est_ghost, p4est_mesh, i, -1, n, 0, NULL, ne, ni);
      if (ni->elem_count > 1)
        printf("%i %i %li strange stuff\n",i,n,ni->elem_count);
      if (ni->elem_count > 0) {
        data->ishell[n] = *((int*)sc_array_index_int(ni,0));
        if (*((int*)sc_array_index_int(ne,0)) >= 0) // Local cell
          data->rshell[n] = this_node;
        else { // in ghost layer
          data->rshell[n] = p4est_mesh->ghost_to_proc[data->ishell[n]];
        }
      }
      sc_array_destroy(ne);
      sc_array_destroy(ni);
    }
    shell.push_back(ls);
  }
    
  //char fname[100];
  //sprintf(fname,"cells_%i.list",this_node);
  //FILE* h = fopen(fname,"w");
  
  // compute ghost, mirror and boundary information
  // here the ghost layer around the local domain is computed
  for (int i=0;i<dd.p4est->local_num_quadrants;++i) {
    p4est_quadrant_t* q = p4est_mesh_get_quadrant(dd.p4est,p4est_mesh,i);
    quad_data_t *data = (quad_data_t*)(q->p.user_data);
    double xyz[3];
    p4est_qcoord_to_vertex(dd.p4est_conn, p4est_mesh->quad_to_tree[i], q->x, q->y, q->z, xyz);
    uint64_t ql = 1<<p4est_tree_array_index(dd.p4est->trees,p4est_mesh->quad_to_tree[i])->maxlevel;
    uint64_t x = xyz[0]*ql;
    uint64_t y = xyz[1]*ql;
    uint64_t z = xyz[2]*ql;
    //fprintf(h,"%i %li %ix%ix%i ",i,dd_p4est_cell_morton_idx(x,y,z), shell[i].coord[0],shell[i].coord[1],shell[i].coord[2]);
    
    // Loop all 27 cells in the fullshell
    for (uint64_t zi=0;zi <= 2;zi++)
      for (uint64_t yi=0;yi <= 2;yi++)
        for (uint64_t xi=0;xi <= 2;xi++) {
          if (xi == 1 && yi == 1 && zi == 1) continue;
          // Check if this node has already been processed using the unique index
          uint64_t qidx = (x+xi) | ((y+yi)<<21) | ((z+zi)<<42);
          size_t pos = 0;
          while (pos < quads.size() && quads[pos] != qidx) ++pos;
          
          if (pos == quads.size()) { // Cell has not been processed yet
            quads.push_back(qidx); // Add it to list
            local_shell_t ls;
            // Copy corresponding information from p4est internal struct
            ls.idx = data->ishell[neighbor_lut[zi][yi][xi]];
            ls.rank = data->rshell[neighbor_lut[zi][yi][xi]];
            ls.shell = 2; // This is a ghost cell, since all locals have been added before
            ls.boundary = 0;
            for (int n=0;n<26;++n) ls.neighbor[n] = -1; // Neighbors from ghosts do not matter
            // Set/Update ghost and corresponding local cells boundary info
            if (PERIODIC(0) && (x + xi) == 0) ls.boundary |= 1;
            if (PERIODIC(0) && (x + xi) == grid_size[0] + 1) ls.boundary |= 2;
            if (PERIODIC(1) && (y + yi) == 0) ls.boundary |= 4;
            if (PERIODIC(1) && (y + yi) == grid_size[1] + 1) ls.boundary |= 8;
            if (PERIODIC(2) && (z + zi) == 0) ls.boundary |= 16;
            if (PERIODIC(2) && (z + zi) == grid_size[2] + 1) ls.boundary |= 32;
            if (xi == 0 && yi == 1 && zi == 1) shell[i].boundary |= 1;
            if (xi == 2 && yi == 1 && zi == 1) shell[i].boundary |= 2;
            if (xi == 1 && yi == 0 && zi == 1) shell[i].boundary |= 4;
            if (xi == 1 && yi == 2 && zi == 1) shell[i].boundary |= 8;
            if (xi == 1 && yi == 1 && zi == 0) shell[i].boundary |= 16;
            if (xi == 1 && yi == 1 && zi == 2) shell[i].boundary |= 32;
            // Link the new cell to a local cell
            shell[i].neighbor[neighbor_lut[zi][yi][xi]] = shell.size();
            ls.coord[0] = int(x + xi) - 1;
            ls.coord[1] = int(y + yi) - 1;
            ls.coord[2] = int(z + zi) - 1;
            ls.p_cnt = 0;
            if (ls.rank == this_node) ls.p_cnt = 1;
            for (uint64_t l=num_local_cells;l<shell.size();++l) {
              if (shell[l].idx == ls.idx && shell[l].rank == ls.rank) {
                if (shell[l].boundary < ls.boundary)
                  ls.p_cnt += 1;
                else
                  shell[l].p_cnt += 1;
              }
            }
            shell.push_back(ls); // add the new ghost cell to all cells
            shell[i].shell = 1; // The cell for which this one was added is at the domain bound
          } else { // Cell already exists in list
            if (shell[pos].shell == 2) { // is it a ghost cell, then ubdate the boundary info
              // of the current local cell, since they are neighbors
              shell[i].shell = 1; // this local cell is at domain boundary
              // Update boundary info
              if (xi == 0 && yi == 1 && zi == 1) shell[i].boundary |= 1;
              if (xi == 2 && yi == 1 && zi == 1) shell[i].boundary |= 2;
              if (xi == 1 && yi == 0 && zi == 1) shell[i].boundary |= 4;
              if (xi == 1 && yi == 2 && zi == 1) shell[i].boundary |= 8;
              if (xi == 1 && yi == 1 && zi == 0) shell[i].boundary |= 16;
              if (xi == 1 && yi == 1 && zi == 2) shell[i].boundary |= 32;
            }
            // Link it as neighbor
            shell[i].neighbor[neighbor_lut[zi][yi][xi]] = pos;
          }
        }
    /*for (int n=0;n<26;++n) {
      if (n < 6)
        fprintf(h,"f%i[",n);
      else if (n < 18)
        fprintf(h,"e%i[",n-6);
      else
        fprintf(h,"c%i[",n-18);
      if (data->ishell[n] >= 0)
        fprintf(h,"%i:%i] ",data->rshell[n],data->ishell[n]);
      else
        fprintf(h,"] ");
    }
    fprintf(h,"\n");*/
  }
  
  //fclose(h);
  
  // Copy the generated data to globals
  num_cells = quads.size();
  num_local_cells = (size_t) dd.p4est->local_num_quadrants;
  num_ghost_cells = num_cells - num_local_cells;

  dd.p4est_shell = std::move(shell);

  CELL_TRACE(printf("%d : %ld, %ld, %ld\n",this_node,num_cells,num_local_cells,num_ghost_cells));

#ifndef P4EST_NOCHANGE  
  // allocate memory
  realloc_cells(num_cells);
  realloc_cellplist(&local_cells, local_cells.n = num_local_cells);
  realloc_cellplist(&ghost_cells, ghost_cells.n = num_ghost_cells);
#endif
  
  dd_p4est_write_vtk();
  
  CELL_TRACE(printf("%d: %.3f %.3fx%.3fx%.3f %.3fx%.3fx%.3f\n",
    this_node, max_range, box_l[0],box_l[1],box_l[2], dd.cell_size[0],dd.cell_size[1],dd.cell_size[2]));

}
//--------------------------------------------------------------------------------------------------

// Compute communication partners and the cells that need to be comunicated
void dd_p4est_comm () {
  CALL_TRACE();
  
  // List of cell idx marked for send/recv for each process
  std::vector<std::vector<int>>      send_idx(n_nodes);
  std::vector<std::vector<int>>      recv_idx(n_nodes);
  // List of all directions for those communicators encoded in a bitmask
  std::vector<std::vector<uint64_t>> send_tag(n_nodes);
  std::vector<std::vector<uint64_t>> recv_tag(n_nodes);
  // Or-Sum (Union) over the lists above
  std::vector<uint64_t> send_cnt_tag(n_nodes, 0UL);
  std::vector<uint64_t> recv_cnt_tag(n_nodes, 0UL);
  // Number of cells for each communication (rank and direction)
  // Default value initialized to all 0
  std::vector<std::array<int, 64>> send_cnt(n_nodes);
  std::vector<std::array<int, 64>> recv_cnt(n_nodes);
  
  // Total number of send and recv
  int num_send = 0;
  int num_recv = 0;
  // Is 1 for a process if there is any communication, 0 if none
  std::vector<int8_t> num_send_flag(n_nodes, 0);
  std::vector<int8_t> num_recv_flag(n_nodes, 0);
  
  // Prepare all lists
  num_comm_proc = 0;

  comm_proc.resize(n_nodes);
  std::fill(std::begin(comm_proc), std::end(comm_proc), -1);

  // create send and receive list
  //char fname[100];
  //sprintf(fname,"cells_conn_%i.list",this_node);
  //FILE* h = fopen(fname,"w");
  // Loop all cells
  for (int i=0;i<num_cells;++i) {
    // is ghost cell that is linked to a process? -> add to recv list
    if (dd.p4est_shell[i].rank >= 0 && dd.p4est_shell[i].shell == 2) {
      int irank = dd.p4est_shell[i].rank;
      int pos = 0;
      // find position to add new element (keep order)
      while (pos < recv_idx[irank].size() && 
        dd.p4est_shell[recv_idx[irank][pos]].idx <= dd.p4est_shell[i].idx) pos++;
      
      if (pos >= recv_idx[irank].size()) { // Add to end of vector
        recv_idx[irank].push_back(i);
        recv_tag[irank].push_back(1L<<dd.p4est_shell[i].boundary);
      // insert if this cell has not been added yet
      } else if (dd.p4est_shell[recv_idx[irank][pos]].idx != dd.p4est_shell[i].idx) {
        recv_idx[irank].insert(recv_idx[irank].begin() + pos, i);
        recv_tag[irank].insert(recv_tag[irank].begin() + pos, 1L<<dd.p4est_shell[i].boundary);
      // update diraction info for communication if already added but for other direction
      } else {
        recv_tag[irank][pos] |= 1L<<dd.p4est_shell[i].boundary;        
      }
      // count what happend above
      recv_cnt[irank][dd.p4est_shell[i].boundary] += 1;
      if ((recv_cnt_tag[irank] & (1L<<dd.p4est_shell[i].boundary)) == 0) {
        ++num_recv;
        recv_cnt_tag[irank] |= 1L<<dd.p4est_shell[i].boundary;
      }
      //recv_cnt[p4est_shell[i].rank].insert(recv_cnt[p4est_shell[i].rank].begin() + pos, i); //p4est_shell[i].idx);
      if (num_recv_flag[irank] == 0) {
          //++num_recv;
          comm_proc[irank] = num_comm_proc;
          num_comm_proc += 1;
          num_recv_flag[irank] = 1;
      }
    }
    // is mirror cell (at domain boundary)? -> add to send list
    if (dd.p4est_shell[i].shell == 1) {
      //for (int n=0;n<n_nodes;++n) comm_cnt[n] = 0;
      // loop fullshell
      for (int n=0;n<26;++n) {
        int nidx = dd.p4est_shell[i].neighbor[n];
        int nrank = dd.p4est_shell[nidx].rank;
        if (nidx < 0 || nrank < 0) continue; // invalid neighbor
        if (dd.p4est_shell[nidx].shell != 2) continue; // no need to send to local cell
        // check if this is the first time to add this mirror cell
        if (!send_tag[nrank].empty() && send_idx[nrank].back() == i) { // already added
          if ((send_tag[nrank].back() & (1L<<dd.p4est_shell[nidx].boundary))) continue;
          // update direction info for this communication
          send_tag[nrank].back() |= (1L<<dd.p4est_shell[nidx].boundary);
        } else { // not added yet -> do so
          send_idx[nrank].push_back(i);
          send_tag[nrank].push_back(1L<<dd.p4est_shell[nidx].boundary);
        }
        // count what happend
        send_cnt[nrank][dd.p4est_shell[nidx].boundary] += 1;
        if ((send_cnt_tag[nrank] & (1L<<dd.p4est_shell[nidx].boundary)) == 0) {
          ++num_send;
          send_cnt_tag[nrank] |= 1L<<dd.p4est_shell[nidx].boundary;
        }
        if (num_send_flag[nrank] == 0) {
          //++num_send;
          num_send_flag[nrank] = 1;
        }
      }
    }
    //fprintf(h,"%i %i:%li (%i) %i %i [ ",i,p4est_shell[i].rank,p4est_shell[i].idx,p4est_shell[i].p_cnt,
    //  p4est_shell[i].shell,p4est_shell[i].boundary);
    //for (int n=0;n<26;++n) fprintf(h,"%i ",p4est_shell[i].neighbor[n]);
    //fprintf(h,"]\n");
  }
  //fclose(h);
  /*sprintf(fname,"send_%i.list",this_node);
  h = fopen(fname,"w");
  for (int n=0;n<n_nodes;++n)
    for (int i=0;i<send_idx[n].size();++i)
      fprintf(h,"%i:%i 0x%016lx\n",n,send_idx[n][i],send_tag[n][i]);
  fclose(h);
  sprintf(fname,"recv_%i.list",this_node);
  h = fopen(fname,"w");
  for (int n=0;n<n_nodes;++n)
    for (int i=0;i<recv_idx[n].size();++i)
      fprintf(h,"%i:%li 0x%016lx\n",n,p4est_shell[recv_idx[n][i]].idx,recv_tag[n][i]);
  fclose(h);*/
  
  // prepare communicator
  CELL_TRACE(fprintf(stdout,"%i : proc %i send %i, recv %i\n",this_node,num_comm_proc,num_send,num_recv));
  num_comm_recv = num_recv;
  num_comm_send = num_send;
  comm_recv.resize(num_recv);
  comm_send.resize(num_send);
  comm_rank.resize(num_comm_proc, COMM_RANK_NONE);

  // Parse all bitmasks and fill the actual lists
  int s_cnt = 0, r_cnt = 0;
  for (int n = 0; n < n_nodes; ++n) {
    if (comm_proc[n] >= 0) comm_rank[comm_proc[n]] = n;
    for (int i=0;i<64;++i) {
      if (num_recv_flag[n] && (recv_cnt_tag[n] & 1L<<i)) {
        comm_recv[r_cnt].cnt = recv_cnt[n][i];
        comm_recv[r_cnt].rank = n;
        comm_recv[r_cnt].dir = i;
        comm_recv[r_cnt].idx.resize(recv_cnt[n][i]);
        for (int j=0,c=0;j<recv_idx[n].size();++j)
          if ((recv_tag[n][j] & (1L<<i)))
            comm_recv[r_cnt].idx[c++] = recv_idx[n][j];
        ++r_cnt;
      }
      if (num_send_flag[n] && (send_cnt_tag[n] & 1L<<i)) {
        comm_send[s_cnt].cnt = send_cnt[n][i];
        comm_send[s_cnt].rank = n;
        comm_send[s_cnt].dir = i;
        comm_send[s_cnt].idx.resize(send_cnt[n][i]);
        for (int j=0,c=0;j<send_idx[n].size();++j)
          if ((send_tag[n][j] & 1L<<i))
            comm_send[s_cnt].idx[c++] = send_idx[n][j];
        ++s_cnt;
      }
    }
  }

  // Debug or rather sanity check
  if (std::find(std::begin(comm_proc), std::end(comm_proc), COMM_RANK_NONE) !=
          std::end(comm_proc) ||
      r_cnt != num_recv || s_cnt != num_send) {
    std::cerr << "[" << this_node << "]"
              << "Error: Comm_recv or comm_send or comm_rank has invalid entry."
              << std::endl;
    errexit();
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part) {
  CALL_TRACE();
  prepare_comm(comm, data_part, num_comm_send + num_comm_recv, true);
  int cnt = 0;
  for (int i=0;i<num_comm_send;++i) {
    comm->comm[cnt].type = GHOST_SEND;
    comm->comm[cnt].node = comm_send[i].rank;
    // The tag distinguishes communications to the same rank
    comm->comm[cnt].tag = comm_send[i].dir;
    comm->comm[cnt].part_lists = (ParticleList**)Utils::malloc(comm_send[i].cnt*sizeof(ParticleList*));
    comm->comm[cnt].n_part_lists = comm_send[i].cnt;
    for (int n=0;n<comm_send[i].cnt;++n)
      comm->comm[cnt].part_lists[n] = &cells[comm_send[i].idx[n]];
    if ((data_part & GHOSTTRANS_POSSHFTD)) {
      // Set shift according to communication direction
      if ((comm_send[i].dir &  1)) comm->comm[cnt].shift[0] =  box_l[0];
      if ((comm_send[i].dir &  2)) comm->comm[cnt].shift[0] = -box_l[0];
      if ((comm_send[i].dir &  4)) comm->comm[cnt].shift[1] =  box_l[1];
      if ((comm_send[i].dir &  8)) comm->comm[cnt].shift[1] = -box_l[1];
      if ((comm_send[i].dir & 16)) comm->comm[cnt].shift[2] =  box_l[2];
      if ((comm_send[i].dir & 32)) comm->comm[cnt].shift[2] = -box_l[2];
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
    comm->comm[cnt].part_lists = (ParticleList**)Utils::malloc(comm_recv[i].cnt*sizeof(ParticleList*));
    comm->comm[cnt].n_part_lists = comm_recv[i].cnt;
    for (int n=0;n<comm_recv[i].cnt;++n)
      comm->comm[cnt].part_lists[n] = &cells[comm_recv[i].idx[n]];
    ++cnt;
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_mark_cells () {
  CALL_TRACE();
  // Take the memory map as they are. First local cells (along Morton curve).
  // This also means cells has the same ordering as p4est_shell
#ifndef P4EST_NOCHANGE
  for (int c=0;c<num_local_cells;++c)
    local_cells.cell[c] = &cells[c];
  // Ghost cells come after local cells
  for (int c=0;c<num_ghost_cells;++c)
    ghost_cells.cell[c] = &cells[num_local_cells + c];
#endif
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_update_comm_w_boxl(GhostCommunicator *comm) {
  CALL_TRACE();
  int cnt = 0;
  for (int i=0;i<num_comm_send;++i) {
    // Reset shift according to communication direction
    comm->comm[cnt].shift[0] = comm->comm[cnt].shift[1] = comm->comm[cnt].shift[2] = 0.0;
    if ((comm_send[i].dir &  1)) comm->comm[cnt].shift[0] =  box_l[0];
    if ((comm_send[i].dir &  2)) comm->comm[cnt].shift[0] = -box_l[0];
    if ((comm_send[i].dir &  4)) comm->comm[cnt].shift[1] =  box_l[1];
    if ((comm_send[i].dir &  8)) comm->comm[cnt].shift[1] = -box_l[1];
    if ((comm_send[i].dir & 16)) comm->comm[cnt].shift[2] =  box_l[2];
    if ((comm_send[i].dir & 32)) comm->comm[cnt].shift[2] = -box_l[2];
    ++cnt;
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_init_cell_interaction() {
#ifndef P4EST_NOCHANGE
  dd.cell_inter = (IA_Neighbor_List*)Utils::realloc(dd.cell_inter,local_cells.n*sizeof(IA_Neighbor_List));
  for (int i=0;i<local_cells.n; i++) { 
    dd.cell_inter[i].nList = NULL; 
    dd.cell_inter[i].n_neighbors=0; 
  }
  
  for (int i=0;i<num_local_cells;++i) {
    dd.cell_inter[i].nList = (IA_Neighbor*)Utils::realloc(dd.cell_inter[i].nList,CELLS_MAX_NEIGHBORS*sizeof(IA_Neighbor));
    
    // Copy info of the local cell itself
    dd.cell_inter[i].nList[0].cell_ind = i;
    dd.cell_inter[i].nList[0].pList = &cells[i];
    init_pairList(&dd.cell_inter[i].nList[0].vList);
    
    // Copy all other cells in half-shell
    for (int n=1;n<CELLS_MAX_NEIGHBORS;++n) {
      dd.cell_inter[i].nList[n].cell_ind = dd.p4est_shell[i].neighbor[half_neighbor_idx[n]];
      dd.cell_inter[i].nList[n].pList = &cells[dd.p4est_shell[i].neighbor[half_neighbor_idx[n]]];
      init_pairList(&dd.cell_inter[i].nList[n].vList);
    }
    
    dd.cell_inter[i].n_neighbors = CELLS_MAX_NEIGHBORS;
  }
#endif
}
//--------------------------------------------------------------------------------------------------
Cell* dd_p4est_position_to_cell_strict(double pos[3]) {
  CALL_TRACE();
  
  // Does the same as dd_p4est_save_position_to_cell but does not extend the local domain
  // by the error bounds

  auto shellidxcomp = [](const local_shell_t& s, int64_t idx) {
    int64_t sidx = dd_p4est_cell_morton_idx(s.coord[0],
                                            s.coord[1],
                                            s.coord[2]);
    return sidx < idx;
  };

  const auto needle = dd_p4est_pos_morton_idx(pos);

  auto local_end = std::begin(dd.p4est_shell) + num_local_cells;
  auto it = std::lower_bound(std::begin(dd.p4est_shell),
                             local_end,
                             needle,
                             shellidxcomp);
  if (it != local_end
        && dd_p4est_cell_morton_idx(it->coord[0],
                                    it->coord[1],
                                    it->coord[2]) == needle)
    return &cells[std::distance(std::begin(dd.p4est_shell), it)];
  else
    return NULL;
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

  return NULL;
}
//--------------------------------------------------------------------------------------------------
Cell* dd_p4est_position_to_cell(double pos[3]) {
  // Accept OOB particles for the sake of IO.
  // If this is used, you need to manually do a global(!) resort afterwards
  // i.e. in Tcl: sort_particles.
  // xTODO: This could be done automatically: Add another flag to cells.hpp:
  // "int force_global_resort" and MPI_Reduce it in cells_resort_particles.
  // Need to clarify first if this has negative implications on the normal
  // function of ESPResSo since this means, oob particles would silently(!)
  // be accepted.
  // Is now implemented as follows: In communication.cpp in mpi_mpiio a
  // global exchange is performed.
  Cell *c = dd_p4est_position_to_cell_strict(pos);

  if (c) {
    return c;
  } else {
    return &cells[0];
  }
}
//--------------------------------------------------------------------------------------------------
// Checks all particles and resorts them to local cells or sendbuffers
// Note: Particle that stay local and are moved to a cell with higher index are touched twice.
// This is not the most efficient way! It would be better to first remember the cell-particle
// index pair of those particles in a vector and move them at the end.
void dd_p4est_fill_sendbuf (ParticleList *sendbuf, std::vector<int> *sendbuf_dyn) {
  double cell_lc[3], cell_hc[3];
  // Loop over all cells and particles
  for (int i=0;i<num_local_cells;++i) {
    Cell* cell = local_cells.cell[i];
    local_shell_t* shell = &dd.p4est_shell[i];
    
    for (int d=0;d<3;++d) {
      cell_lc[d] = dd.cell_size[d]*(double)shell->coord[d];
      cell_hc[d] = cell_lc[d] + dd.cell_size[d];
      if ((shell->boundary & (1<<(2*d)))) cell_lc[d] -= 0.5*ROUND_ERROR_PREC*box_l[d];
      if ((shell->boundary & (2<<(2*d)))) cell_hc[d] += 0.5*ROUND_ERROR_PREC*box_l[d];
    }
    
    for (int p=0;p<cell->n;++p) {
      Particle* part = &cell->part[p];
      int x,y,z;
      
      // Check if particle has left the cell. (The local domain is extenden by half round error)
      if (part->r.p[0] < cell_lc[0]) x = 0;
      else if (part->r.p[0] >= cell_hc[0]) x = 2;
      else x = 1;
      if (part->r.p[1] < cell_lc[1]) y = 0;
      else if (part->r.p[1] >= cell_hc[1]) y = 2;
      else y = 1;
      if (part->r.p[2] < cell_lc[2]) z = 0;
      else if (part->r.p[2] >= cell_hc[2]) z = 2;
      else z = 1;
      
      int nidx = neighbor_lut[z][y][x];
      if (nidx != -1) { // Particle p outside of cell i
        // recalculate neighboring cell to prevent rounding errors
        // If particle left local domain, check for correct ghost cell, thus without ROUND_ERROR_PREC
        for (int d=0;d<3;++d) {
          cell_lc[d] = dd.cell_size[d]*(double)shell->coord[d];
          cell_hc[d] = cell_lc[d] + dd.cell_size[d];
        }
        if (part->r.p[0] < cell_lc[0]) x = 0;
        else if (part->r.p[0] >= cell_hc[0]) x = 2;
        else x = 1;
        if (part->r.p[1] < cell_lc[1]) y = 0;
        else if (part->r.p[1] >= cell_hc[1]) y = 2;
        else y = 1;
        if (part->r.p[2] < cell_lc[2]) z = 0;
        else if (part->r.p[2] >= cell_hc[2]) z = 2;
        else z = 1;
        
        nidx = neighbor_lut[z][y][x];
        // get neighbor cell
        nidx = shell->neighbor[nidx];
        if (nidx >= num_local_cells) { // Remote Cell (0:num_local_cells-1) -> local, other: ghost
          if (dd.p4est_shell[nidx].rank >= 0) { // This ghost cell is linked to a process
            CELL_TRACE(fprintf(stderr,"%d: dd_ex_and_sort_p: send part %d\n",this_node,part->p.identity));
              
            if (dd.p4est_shell[nidx].rank != this_node) { // It is a remote process
              // copy data to sendbuf according to rank
              int li = comm_proc[dd.p4est_shell[nidx].rank];
              sendbuf_dyn[li].insert(sendbuf_dyn[li].end(), part->bl.e, part->bl.e + part->bl.n);
#ifdef EXCLUSIONS
              sendbuf_dyn[li].insert(sendbuf_dyn[li].end(), part->el.e, part->el.e + part->el.n);
#endif
              int pid = part->p.identity;
              //fold_position(part->r.p, part->l.i);
              move_indexed_particle(&sendbuf[li], cell, p);
              local_particles[pid] = NULL;
              if(p < cell->n) p -= 1;
            } else { // particle stays local, but since it went to a ghost it has to be folded
              fold_position(part->r.p, part->l.i);
              move_indexed_particle(&cells[dd.p4est_shell[nidx].idx], cell, p);
              if(p < cell->n) p -= 1;
            }
          } else { // Particle left global domain and is not tracked by any process anymore
            runtimeErrorMsg() << "particle " << p << " on process " << this_node << " is OB";
            fprintf(stderr, "%i : part %i cell %i is OB [%lf %lf %lf]\n", this_node, i, p, part->r.p[0], part->r.p[1], part->r.p[2]);
          }
        } else { // Local Cell, just move the partilce
          move_indexed_particle(&cells[nidx], cell, p);
          if(p < cell->n) p -= 1;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------------------------------
static int dd_async_exchange_insert_particles(ParticleList *recvbuf, int global_flag, int from) {
  // add all particle in a recvbuf to the local storage
  int dynsiz = 0;

  update_local_particles(recvbuf);

  for (int p = 0; p < recvbuf->n; ++p) {
    double op[3] = {recvbuf->part[p].r.p[0], recvbuf->part[p].r.p[1], recvbuf->part[p].r.p[2]};
    // Sender folds in global case to get the correct receiver rank.
    // Do not undo this folding here.
    if (!global_flag)
      fold_position(recvbuf->part[p].r.p, recvbuf->part[p].l.i);

    dynsiz += recvbuf->part[p].bl.n;
#ifdef EXCLUSIONS
    dynsiz += recvbuf->part[p].el.n;
#endif
  //}
  // Fold direction of dd_append_particles unused.
  
  //for (int p=0;p<recvbuf->n;++p) {
    Cell* target = dd_p4est_save_position_to_cell(recvbuf->part[p].r.p);
    if (target) {
      append_indexed_particle(target, &recvbuf->part[p]);
    } else {
      fprintf(stderr, "proc %i received remote particle p%i out of domain, global %i from proc %i\n\t%lfx%lfx%lf, glob morton idx %li, pos2proc %i\n\told pos %lfx%lfx%lf\n", 
        this_node, recvbuf->part[p].p.identity, global_flag, from,
        recvbuf->part[p].r.p[0], recvbuf->part[p].r.p[1], recvbuf->part[p].r.p[2], 
        dd_p4est_pos_morton_idx(recvbuf->part[p].r.p), dd_p4est_pos_to_proc(recvbuf->part[p].r.p),
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
      alloc_intlist(&p->bl, p->bl.n);
      // used to be memmove, but why?
      memcpy(p->bl.e, &dynrecv[read], p->bl.n * sizeof(int));
      read += p->bl.n;
    } else {
      p->bl.e = NULL;
    }
#ifdef EXCLUSIONS
    if (p->el.n > 0) {
      alloc_intlist(&p->el, p->el.n);
      // used to be memmove, but why?
      memcpy(p->el.e, &dynrecv[read], p->el.n*sizeof(int));
      read += p->el.n;
    }
    else {
      p->el.e = NULL;
    }
#endif
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_exchange_and_sort_particles() {
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
      local_particles[sendbuf[i].part[p].p.identity] = NULL;
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
      int rank = dd_p4est_pos_to_proc(part->r.p);
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
        local_particles[pid] = NULL;
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
      local_particles[sendbuf[i].part[p].p.identity] = NULL;
      free_particle(&sendbuf[i].part[p]);
    }
    realloc_particlelist(&sendbuf[i], 0);
    realloc_particlelist(&recvbuf[i], 0);
  }

#if 0
  // check if received particles acually belong here
  for (int i=0;i<num_local_cells;++i) {
    for (int p = 0; p < cells[i].n; p++) {
      Particle *part = &cells[i].part[p];
      if (dd_p4est_pos_to_proc(part->r.p) != this_node) {
        fprintf(stderr,"W %i:%i : %lfx%lfx%lf %i %s\n", this_node, i,
          part->r.p[0], part->r.p[1], part->r.p[2],
          dd_p4est_pos_to_proc(part->r.p), (dd_p4est_save_position_to_cell(part->r.p)?"l":"r"));
      }
    }
  }
#endif
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_repart_exchange_part (CellPList *old) {
  std::vector<int> send_quads(n_nodes);
  std::vector<int> send_prefix(n_nodes + 1);
  std::vector<int> recv_quads(n_nodes);
  std::vector<int> recv_prefix(n_nodes + 1);
  
  std::vector<std::vector<int>> send_num_part(n_nodes);
  std::vector<std::vector<int>> recv_num_part(n_nodes);
  
  int lb_old_local = old_global_first_quadrant[this_node];
  int ub_old_local = old_global_first_quadrant[this_node + 1];
  int lb_new_local = dd.p4est->global_first_quadrant[this_node];
  int ub_new_local = dd.p4est->global_first_quadrant[this_node + 1];
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
      fprintf(stderr, "[%i] : recv %i (%i)\n", this_node, p, REP_EX_CNT_TAG);
    }
  }

  // send respective quadrants to other processors
  send_prefix[0] = 0;
  int c_cnt = 0;
  for (int p = 0; p < n_nodes; ++p) {
    lb_new_remote = ub_new_remote;
    ub_new_remote = dd.p4est->global_first_quadrant[p + 1];

    send_quads[p] = std::max(0,
                           std::min(ub_old_local, ub_new_remote) -
                               std::max(lb_old_local, lb_new_remote));
    send_prefix[p+1] = send_prefix[p] + send_quads[p];
    
    // Fill send list for number of particles per cell
    send_num_part[p].resize(send_quads[p]);
    init_particlelist(&sendbuf[p]);
    int send_sum = 0;
    for (int c = 0; c < send_quads[p]; ++c) {
      if (p == this_node) {
        recv_num_part[p][c] = old->cell[c_cnt]->n;
        realloc_particlelist(&cells[recv_prefix[p] + c], old->cell[c_cnt]->n);
        //cells[recv_prefix[p] + c].n = old->cell[c_cnt]->n;
      } else {
        realloc_particlelist(&sendbuf[p], sendbuf[p].n + old->cell[c_cnt]->n);
        //sendbuf[p].n = old->cell[c_cnt]->n;
      }
      send_num_part[p][c] = old->cell[c_cnt]->n;
      send_sum += old->cell[c_cnt]->n;
      for (int i = 0; i < old->cell[c_cnt]->n; i++) {
        Particle *part = &old->cell[c_cnt]->part[i];
        if (p != this_node) {
          // It is actually a remote particle -> copy all data to sendbuffer
          sendbuf_dyn[p].insert(sendbuf_dyn[p].end(), part->bl.e,
                                part->bl.e + part->bl.n);
#ifdef EXCLUSIONS
          sendbuf_dyn[p].insert(sendbuf_dyn[p].end(), part->el.e,
                                part->el.e + part->el.n);
#endif
          int pid = part->p.identity;
          //memcpy(&sendbuf[p].part[i], part, sizeof(Particle));
          append_unindexed_particle(&sendbuf[p], part);
          local_particles[pid] = NULL;
        } else { // Particles that stay local
          int pid = part->p.identity;
          memcpy(&cells[recv_prefix[p] + c].part[i], part, sizeof(Particle));
          local_particles[pid] = &cells[recv_prefix[p] + c].part[i];
          cells[recv_prefix[p] + c].n += 1;
        }
      }
      ++c_cnt;
    }
    if (send_sum != sendbuf[p].n) {
      fprintf(stderr, "[%i] send buffer mismatch for process %i\n", this_node, p);
      errexit();
    }
    if (p != this_node && send_quads[p] > 0) {
      MPI_Isend(send_num_part[p].data(),
                send_quads[p], MPI_INT, p, REP_EX_CNT_TAG,
                comm_cart, &sreq[p]);
      fprintf(stderr, "[%i] : send %i (%i)\n", this_node, p, REP_EX_CNT_TAG);
      if (sendbuf[p].n > 0) {
        MPI_Isend(sendbuf[p].part, 
                  sendbuf[p].n * sizeof(Particle), MPI_BYTE, p, REP_EX_PART_TAG,
                  comm_cart, &sreq[p + n_nodes]);
        fprintf(stderr, "[%i] : send %i (%i)\n", this_node, p, REP_EX_PART_TAG);
        if (sendbuf_dyn[p].size() > 0) {
          MPI_Isend(sendbuf_dyn[p].data(), 
                    sendbuf_dyn[p].size(), MPI_INT, p, REP_EX_DYN_TAG,
                    comm_cart, &sreq[p + 2 * n_nodes]);
        fprintf(stderr, "[%i] : send %i (%i)\n", this_node, p, REP_EX_DYN_TAG);
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
      
    fprintf(stderr, "[%i] %i(%i)\n", this_node, source, tag);
      
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
          fprintf(stderr, "[%i] : recv %i (%i)\n", this_node, source, REP_EX_PART_TAG);
          commstat.next(recvidx);
        }
      }
      break;
    case CommunicationStatus::ReceiveStatus::RECV_PARTICLES:
      DIE_IF_TAG_MISMATCH(tag, REP_EX_PART_TAG, "Repart exchange particles");
      dyndatasiz = 0;
      read = 0;
      for (int c = 0; c < recv_quads[source]; ++c) {
        realloc_particlelist(&cells[recv_prefix[source] + c], recv_num_part[source][c]);
        //cells[recv_prefix[source] + c].n = recv_num_part[source][c];
        for (int p = 0; p < recv_num_part[source][c]; ++p) {
          //memcpy(&cells[recv_prefix[source] + c].part[p], &recvbuf[source].part[p], sizeof(Particle));
          append_indexed_particle(&cells[recv_prefix[source] + c], &recvbuf[source].part[read + p]);
          dyndatasiz += recvbuf[source].part[read + p].bl.n;
#ifdef EXCLUSIONS
          dyndatasiz += recvbuf[source].part[read + p].el.n;
#endif
        }
        read += recv_num_part[source][c];
      }
      if (dyndatasiz > 0) {
        recvbuf_dyn[source].resize(dyndatasiz);
        MPI_Irecv(recvbuf_dyn[source].data(), dyndatasiz, MPI_INT, source,
                  REP_EX_DYN_TAG, comm_cart, &rreq[recvidx]);
        fprintf(stderr, "[%i] : send %i (%i)\n", this_node, source, REP_EX_DYN_TAG);
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
            alloc_intlist(&p->bl, p->bl.n);
            memcpy(p->bl.e, &recvbuf_dyn[source][read], p->bl.n * sizeof(int));
            read += p->bl.n;
          } else {
            p->bl.e = NULL;
          }
#ifdef EXCLUSIONS
          if (p->el.n > 0) {
            alloc_intlist(&p->el, p->el.n);
            memcpy(p->el.e, &recvbuf_dyn[source][read], p->el.n*sizeof(int));
            read += p->el.n;
          } else {
            p->el.e = NULL;
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
      local_particles[sendbuf[i].part[p].p.identity] = NULL;
      free_particle(&sendbuf[i].part[p]);
    }
    realloc_particlelist(&sendbuf[i], 0);
    realloc_particlelist(&recvbuf[i], 0);
  }
}
//--------------------------------------------------------------------------------------------------
// Maps a position to the cartesian grid and returns the morton index of this coordinates
// Note: the global morton index returned here is NOT equal to the local cell index!!!
int64_t dd_p4est_pos_morton_idx(double pos[3]) {
  double pfold[3] = {pos[0], pos[1], pos[2]};
  int im[3] = {0, 0, 0}; /* dummy */
  fold_position(pfold, im);
  for (int d=0;d<3;++d) {
    double errmar = 0.5*ROUND_ERROR_PREC * box_l[d];
    if (pos[d] < 0 && pos[d] > -errmar) pfold[d] = 0;
    else if (pos[d] >= box_l[d] && pos[d] < box_l[d] + errmar) pfold[d] = pos[d] - 0.5*dd.cell_size[d];
    // In the other two cases ("pos[d] <= -errmar" and
    // "pos[d] >= box_l[d] + errmar") pfold is correct.
  }
  return dd_p4est_cell_morton_idx(pfold[0] * dd.inv_cell_size[0], 
    pfold[1] * dd.inv_cell_size[1], pfold[2] * dd.inv_cell_size[2]);
}
//--------------------------------------------------------------------------------------------------
// Find the process that handles the position
int dd_p4est_pos_to_proc(double pos[3]) {
  auto it = std::upper_bound(
      std::begin(p4est_space_idx), std::end(p4est_space_idx) - 1,
      dd_p4est_pos_morton_idx(pos), [](int i, int64_t idx) { return i < idx; });

  return std::distance(std::begin(p4est_space_idx), it) - 1;
}
//--------------------------------------------------------------------------------------------------
// Repartitions the given p4est, such that process boundaries do not intersect the partition of the
// p4est_dd grid. This only works if both grids have a common p4est_connectivity and the given forest
// is on the same or finer level.
void dd_p4est_partition(p4est_t *p4est, p4est_mesh_t *mesh, p4est_connectivity_t *conn) {
  p4est_locidx_t num_quad_per_proc[n_nodes];
  p4est_locidx_t num_quad_per_proc_global[n_nodes];
  for (int i=0;i<n_nodes;++i) num_quad_per_proc[i] = 0;
  
  // Check for each of the quadrants of the given p4est, to which MD cell it maps
  for (int i=0;i<p4est->local_num_quadrants;++i) {
    p4est_quadrant_t *q = p4est_mesh_get_quadrant(p4est,mesh,i);
    double xyz[3];
    p4est_qcoord_to_vertex(conn,mesh->quad_to_tree[i],q->x,q->y,q->z,xyz);
    /*int64_t idx = dd_p4est_cell_morton_idx(xyz[0]*(1<<grid_level),
                                           xyz[1]*(1<<grid_level),xyz[2]*(1<<grid_level));
                                           //dd_p4est_pos_morton_idx(xyz);
    for (int n=1;n<=n_nodes;++n) {
      if (p4est_space_idx[n] > idx) {
        num_quad_per_proc[n - 1] += 1;
        break;
      }
    }*/
    num_quad_per_proc[dd_p4est_pos_to_proc(xyz)] += 1;
  }
  CELL_TRACE(printf("%i : repartition %i cells: ", this_node, p4est->local_num_quadrants));
  for (int i=0;i<n_nodes;++i) {
    CELL_TRACE(printf("%i ",num_quad_per_proc[i]));
  }
  
  CELL_TRACE(printf("\n"));
  
  // Geather this information over all processes
  MPI_Allreduce(num_quad_per_proc, num_quad_per_proc_global, n_nodes, P4EST_MPI_LOCIDX, MPI_SUM, comm_cart);
  
  p4est_locidx_t sum = 0;
  for (int i=0;i<n_nodes;++i) sum += num_quad_per_proc_global[i];
  if (sum < p4est->global_num_quadrants) {
    printf("%i : quadrants lost while partitioning\n", this_node);
    return;
  }
  
  CELL_TRACE(printf("%i : repartitioned LB %i\n", this_node, num_quad_per_proc_global[this_node]));
  
  // Repartition with the computed distribution
  p4est_partition_given(p4est, num_quad_per_proc_global);
}
//--------------------------------------------------------------------------------------------------
// This is basically a copy of the dd_on_geometry_change
void dd_p4est_on_geometry_change(int flags) {
  cells_re_init(CELL_STRUCTURE_CURRENT);
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_write_particle_vtk(char *filename) {
  // strip file endings
  char *pos_file_ending = strrchr(filename, '.');
  if (pos_file_ending != 0) {
    *pos_file_ending = '\0';
  } else {
    pos_file_ending = strpbrk(filename, "\0");
  }
  char fname[1024];
  // node 0 writes the header file
  if (this_node == 0) {
    sprintf(fname,"%s.pvtp",filename);
    FILE *h = fopen(fname, "w");
    fprintf(h,"<?xml version=\"1.0\"?>\n");
    fprintf(h,"<VTKFile type=\"PPolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n\t");
    fprintf(h,"<PPolyData GhostLevel=\"0\">\n\t\t<PPoints>\n\t\t\t");
    fprintf(h,"<PDataArray type=\"Float32\" Name=\"Position\" NumberOfComponents=\"3\" format=\"ascii\"/>\n");
    fprintf(h,"\t\t</PPoints>\n\t\t");
    fprintf(h,"<PPointData Scalars=\"mpirank,part_id,cell_id\" Vectors=\"velocity\">\n\t\t\t");
    fprintf(h,"<PDataArray type=\"Int32\" Name=\"mpirank\" format=\"ascii\"/>\n\t\t\t");
    fprintf(h,"<PDataArray type=\"Int32\" Name=\"part_id\" format=\"ascii\"/>\n\t\t\t");
    fprintf(h,"<PDataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\"/>\n\t\t\t");
    fprintf(h,"<PDataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\"/>\n\t\t");
    fprintf(h,"</PPointData>\n");
    for (int p=0;p<n_nodes;++p)
      fprintf(h,"\t\t<Piece Source=\"%s_%04i.vtp\"/>\n",filename, p);
    fprintf(h,"\t</PPolyData>\n</VTKFile>\n");
    fclose(h);
  }
  // write the actual parallel particle files
  sprintf(fname,"%s_%04i.vtp",filename,this_node);
  int num_p=0;
  for (int c = 0; c < local_cells.n; c++) {
    num_p += local_cells.cell[c]->n;
  }
  FILE *h = fopen(fname, "w");
  fprintf(h,"<?xml version=\"1.0\"?>\n");
  fprintf(h,"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n\t");
  fprintf(h,"<PolyData>\n\t\t<Piece NumberOfPoints=\"%i\" NumberOfVerts=\"0\" ",num_p);
  fprintf(h,"NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n\t\t\t<Points>\n\t\t\t\t");
  fprintf(h,"<DataArray type=\"Float32\" Name=\"Position\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p=0;p<np;++p) {
      fprintf(h,"\t\t\t\t\t%le %le %le\n",part[p].r.p[0],part[p].r.p[1],part[p].r.p[2]);
    }
  }
  fprintf(h,"\t\t\t\t</DataArray>\n\t\t\t</Points>\n\t\t\t");
  fprintf(h,"<PointData Scalars=\"mpirank,part_id,cell_id\" Vectors=\"velocity\">\n\t\t\t\t");
  fprintf(h,"<DataArray type=\"Int32\" Name=\"mpirank\" format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    for (int p=0;p<np;++p) {
      fprintf(h,"%i ",this_node);
    }
  }
  fprintf(h,"\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Int32\" Name=\"part_id\" format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p=0;p<np;++p) {
      fprintf(h,"%i ",part[p].p.identity);
    }
  }
  fprintf(h,"\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\">\n\t\t\t\t\t");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    for (int p=0;p<np;++p) {
      fprintf(h,"%i ",c);
    }
  }
  fprintf(h,"\n\t\t\t\t</DataArray>\n\t\t\t\t<DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (int c = 0; c < local_cells.n; c++) {
    int np = local_cells.cell[c]->n;
    Particle *part = local_cells.cell[c]->part;
    for (int p=0;p<np;++p) {
      fprintf(h,"\t\t\t\t\t%le %le %le\n",
        part[p].m.v[0]/time_step,part[p].m.v[1]/time_step,part[p].m.v[2]/time_step);
    }
  }
  fprintf(h,"\t\t\t\t</DataArray>\n\t\t\t</PointData>\n");
  fprintf(h,"\t\t</Piece>\n\t</PolyData>\n</VTKFile>\n");
  fclose(h);
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_write_vtk() {
  // write cells to VTK with a given streching
  //p4est_vtk_write_file_scale (p4est, NULL, P4EST_STRING "_dd", box_l[0]/(double)brick_size[0]);
  
  /*char fname[100];
  sprintf(fname,"cells_conn_%i.list",this_node);
  FILE* h = fopen(fname,"w");
  for (int i=0;i<num_cells;++i) {
    fprintf(h,"%i %i:%li (%i) %i %i [ ",i, p4est_shell[i].rank, p4est_shell[i].idx,
      p4est_shell[i].p_cnt, p4est_shell[i].shell, p4est_shell[i].boundary);
    for (int n=0;n<26;++n) fprintf(h,"%i ",p4est_shell[i].neighbor[n]);
    fprintf(h,"]\n");
  }
  fclose(h);*/
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

  part_nquads.resize(n_nodes);
  std::fill(part_nquads.begin(), part_nquads.end(),
            static_cast<p4est_locidx_t>(0));
  // Determine prefix and target load
  double localsum = std::accumulate(metric.begin(), metric.end(), 0.0);
  double sum, prefix = 0; // Initialization is necessary on rank 0!
  MPI_Allreduce(&localsum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
  MPI_Exscan(&localsum, &prefix, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
  double target = sum / n_nodes;

  if (debug) {
    printf("[%i] NCells: %zu\n", this_node, num_local_cells);
    printf("[%i] Local : %lf\n", this_node, localsum);
    printf("[%i] Global: %lf\n", this_node, sum);
    printf("[%i] Target: %lf\n", this_node, target);
    printf("[%i] Prefix: %lf\n", this_node, prefix);
  }

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

  // TODO: Could try to steal quads from neighbors.
  //       Global reshifting (i.e. stealing from someone else than the direct
  //       neighbors) is not a good idea since it globally changes the metric.
  //       Anyways, this is most likely due to a bad quad/proc quotient.
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
      if (dd.p4est->global_num_quadrants != totnquads) {
        fprintf(stderr,
                "[%i] ERROR: totnquads = %li but global_num_quadrants = %li\n",
                this_node, totnquads, dd.p4est->global_num_quadrants);
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
p4est_dd_repartition(const std::string& desc, bool debug)
{
  if (desc == "statistics") {
    repart::print_cell_info("!>>> Statistics", "none");
    return;
  }

  std::vector<double> weights = repart::metric{desc}();
  p4est_dd_repart_calc_nquads(weights, debug);

  repart::print_cell_info("!>>> Before Repart", desc);
  cells_re_init(CELL_STRUCTURE_CURRENT, true);
  repart::print_cell_info("!>>> After Repart", desc);
}

#endif

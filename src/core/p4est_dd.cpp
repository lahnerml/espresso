#include "p4est_dd.hpp"
//--------------------------------------------------------------------------------------------------
#ifdef DD_P4EST
//--------------------------------------------------------------------------------------------------
#ifdef LEES_EDWARDS
#error "p4est and Lees-Edwards are not compatible yet."
#endif // LEES_EDWARDS
//--------------------------------------------------------------------------------------------------
#include "call_trace.hpp"
#include "domain_decomposition.hpp"
#include "ghosts.hpp"
#include "p4est_utils.hpp"

#ifdef LB_ADAPTIVE
#include "lb-adaptive.hpp"
#endif // LB_ADAPTIVE

#include <p4est_to_p8est.h>
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_vtk.h>
#include <vector>
#include <tuple>

// Fast to compare 3d index
typedef uint64_t FastIndex3D;
static inline FastIndex3D idx3d(uint64_t x, uint64_t y, uint64_t z) {
  return x | (y << 21) | (z << 42);
}

//--------------------------------------------------------------------------------------------------
#define CELLS_MAX_NEIGHBORS 14
//--------------------------------------------------------------------------------------------------
static int brick_size[3]; // number of trees in each direction
static int grid_size[3];  // number of quadrants/cells in each direction
static int grid_level = 0;
//--------------------------------------------------------------------------------------------------
static std::vector<comm_t>
    comm_send; // Internal send lists, idx maps to local cells
static std::vector<comm_t>
    comm_recv;                // Internal recv lists, idx maps to ghost cells
static int num_comm_send = 0; // Number of send lists
static int num_comm_recv =
    0; // Number of recc lists (should be euqal to send lists)
static std::vector<int> comm_proc; // Number of communicators per Process
static std::vector<int>
    comm_rank; // Communication partner index for a certain communication
static int num_comm_proc = 0; // Total Number of bidirectional communications
//--------------------------------------------------------------------------------------------------
static size_t num_cells = 0;
static size_t num_local_cells = 0;
static size_t num_ghost_cells = 0;
//--------------------------------------------------------------------------------------------------
int dd_p4est_num_trees_in_dir(int d) {
  if (d < 0 || d > 2) {
    fprintf(stderr, "%s called with dimension < 0 or > 2.\n", __FUNCTION__);
    errexit();
  }
  return brick_size[d];
}
//--------------------------------------------------------------------------------------------------
static const int neighbor_lut[3][3][3] =
    { // Mapping from [z][y][x] to p4est neighbor index
        {{18, 6, 19}, {10, 4, 11}, {20, 7, 21}},
        {{14, 2, 15}, {0, -1, 1}, {16, 3, 17}}, //[1][1][1] is the cell itself
        {{22, 8, 23}, {12, 5, 13}, {24, 9, 25}}};
//--------------------------------------------------------------------------------------------------
static const int half_neighbor_idx[14] =
    { // p4est neighbor index of halfshell [0] is cell itself
        -1, 1, 16, 3, 17, 22, 8, 23, 12, 5, 13, 24, 9, 25};
//--------------------------------------------------------------------------------------------------
// Internal p4est structure
typedef struct {
  int rank;       // the node id
  int lidx;       // local index
  int ishell[26]; // local or ghost index of fullshell in p4est
  int rshell[26]; // rank of fullshell cells
} quad_data_t;
//--------------------------------------------------------------------------------------------------
int dd_p4est_full_shell_neigh(int cell, int neighidx) {
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
static void init_fn(p4est_t *p4est, p4est_topidx_t tree, p4est_quadrant_t *q) {
  quad_data_t& qd = *static_cast<quad_data_t *>(q->p.user_data);
  qd.rank = this_node;
  qd.lidx = -1;
  std::fill(std::begin(qd.ishell), std::end(qd.ishell), -1);
  std::fill(std::begin(qd.rshell), std::end(qd.rshell), -1);
}
//--------------------------------------------------------------------------------------------------
static inline int count_trailing_zeros(int x) {
  int z = 0;
  for (; (x & 1) == 0; x >>= 1)
    z++;
  return z;
}

// Compute the grid- and bricksize according to box_l and maxrange
int dd_p4est_cellsize_optimal() {
  int ncells[3] = {1, 1, 1};

  // compute number of cells
  if (max_range > ROUND_ERROR_PREC * box_l[0]) {
    std::transform(std::begin(box_l), std::end(box_l), std::begin(ncells),
                   [](double box){ return std::max<int>(box / max_range, 1);});
  }

  std::copy(std::begin(ncells), std::end(ncells), std::begin(grid_size));

  // divide all dimensions by biggest common power of 2
  int lvl = count_trailing_zeros(ncells[0] | ncells[1] | ncells[2]);

  std::transform(std::begin(ncells), std::end(ncells), std::begin(brick_size),
                 [lvl](int nc){ return nc >> lvl; });

  return lvl; // return the level of the grid
}
//--------------------------------------------------------------------------------------------------
// Creates a forest with box_l trees in each direction
int dd_p4est_cellsize_even() {
#ifdef LB_ADAPTIVE
  P4EST_ASSERT(0 <= max_range && max_range <= 1.0);
#endif // LB_ADAPTIVE

  std::transform(std::begin(box_l), std::end(box_l), std::begin(brick_size),
                 [](double box){ return static_cast<int>(box); });

  int ncells = 1;
  if (max_range > ROUND_ERROR_PREC * box_l[0])
    ncells = std::max<int>(1.0 / max_range, 1);

  int lvl = Utils::nat_log2_floor(ncells);

  std::transform(std::begin(brick_size), std::end(brick_size),
                 std::begin(grid_size), [lvl](int bs){ return bs << lvl; });

  return lvl; // Return level > 0 if max_range <= 0.5
}
//--------------------------------------------------------------------------------------------------

enum class Direction { x_l = 1, x_r = 2, y_l = 4, y_r = 8, z_l = 16, z_r = 32 };

/** Returns the boundary flags for an inner cell given a ghost offset
 * Given that (x, y, z) is no ghost and (x+xi, y+yi, z+zi) is ghost,
 * determines the boundary flags for cell (x, y, z) from the offsets
 * (xi, yi, zi).
 */
int dd_p4est_boundary_flags_for_neighbor(int xi, int yi, int zi) {
  int res = 0;
  if (xi == 0 && yi == 1 && zi == 1)
    res |= static_cast<int>(Direction::x_l);
  if (xi == 2 && yi == 1 && zi == 1)
    res |= static_cast<int>(Direction::x_r);
  if (xi == 1 && yi == 0 && zi == 1)
    res |= static_cast<int>(Direction::y_l);
  if (xi == 1 && yi == 2 && zi == 1)
    res |= static_cast<int>(Direction::y_r);
  if (xi == 1 && yi == 1 && zi == 0)
    res |= static_cast<int>(Direction::z_l);
  if (xi == 1 && yi == 1 && zi == 2)
    res |= static_cast<int>(Direction::z_r);
  return res;
}

/** Returns boundary flags for cells of index (x,z,y).
 */
int dd_p4est_boundary_flags(int x, int y, int z)
{
  int res = 0;
  if (PERIODIC(0) && x == 0)
    res |= static_cast<int>(Direction::x_l);
  if (PERIODIC(0) && x == grid_size[0] + 1)
    res |= static_cast<int>(Direction::x_r);
  if (PERIODIC(1) && y == 0)
    res |= static_cast<int>(Direction::y_l);
  if (PERIODIC(1) && y == grid_size[1] + 1)
    res |= static_cast<int>(Direction::y_r);
  if (PERIODIC(2) && z == 0)
    res |= static_cast<int>(Direction::z_l);
  if (PERIODIC(2) && z == grid_size[2] + 1)
    res |= static_cast<int>(Direction::z_r);
  return res;
}

void dd_p4est_create_grid() {
  // printf("%i : new MD grid\n", this_node);
  CALL_TRACE();
  // Clear old data to prevent it from being used accidentally
  comm_rank.clear();
  comm_proc.clear();
  comm_recv.clear();
  comm_send.clear();
  dd.p4est_shell.clear();
#ifdef LB_ADAPTIVE
  // the adaptive LB has a strange grid, thus we have to do something similar
  // here
  if (max_range < 1.0)
    grid_level = dd_p4est_cellsize_even();
  else
    grid_level = dd_p4est_cellsize_optimal();
#else
  grid_level = dd_p4est_cellsize_optimal();
#endif

#ifndef P4EST_NOCHANGE
  // set global variables
  dd.cell_size[0] = box_l[0] / (double)grid_size[0];
  dd.cell_size[1] = box_l[1] / (double)grid_size[1];
  dd.cell_size[2] = box_l[2] / (double)grid_size[2];
  dd.inv_cell_size[0] = 1.0 / dd.cell_size[0];
  dd.inv_cell_size[1] = 1.0 / dd.cell_size[1];
  dd.inv_cell_size[2] = 1.0 / dd.cell_size[2];
  max_skin =
      std::min(std::min(dd.cell_size[0], dd.cell_size[1]), dd.cell_size[2]) -
      max_cut;

  CELL_TRACE(printf("%i : gridsize %ix%ix%i\n", this_node, grid_size[0],
                    grid_size[1], grid_size[2]));
  CELL_TRACE(printf("%i : bricksize %ix%ix%i level %i\n", this_node,
                    brick_size[0], brick_size[1], brick_size[2], grid_level));
  CELL_TRACE(printf("%i : cellsize %lfx%lfx%lf\n", this_node, dd.cell_size[0],
                    dd.cell_size[1], dd.cell_size[2]));
#endif

  // create p4est structs
  {
    // Save old connectivity beucase p4est_destroy needs it.
    auto oldconn = std::move(dd.p4est_conn);
    dd.p4est_conn =
        std::unique_ptr<p4est_connectivity_t>(p8est_connectivity_new_brick(
            brick_size[0], brick_size[1], brick_size[2], PERIODIC(0),
            PERIODIC(1), PERIODIC(2)));
    dd.p4est = std::unique_ptr<p4est_t>(
        p4est_new_ext(comm_cart, dd.p4est_conn, 0, grid_level, true,
                      sizeof(quad_data_t), init_fn, NULL));
  }
  p4est_partition(dd.p4est, 0, NULL);

  std::vector<p4est_t *> forests = {dd.p4est};
#ifdef LB_ADAPTIVE
  if (lb_p8est != 0) {
    forests.push_back(lb_p8est);
  }
#endif // LB_ADAPTIVE
  p4est_utils_prepare(forests);

  castable_unique_ptr<p4est_ghost_t> p4est_ghost =
      p4est_ghost_new(dd.p4est, P8EST_CONNECT_CORNER);
  castable_unique_ptr<p4est_mesh_t> p4est_mesh =
      p4est_mesh_new_ext(dd.p4est, p4est_ghost, 1, 1, 0, P8EST_CONNECT_CORNER);

  CELL_TRACE(printf("%i : %i %i-%i %i\n", this_node, periodic,
                    p4est->first_local_tree, p4est->last_local_tree,
                    p4est->local_num_quadrants));


  // gather cell neighbors
  std::vector<FastIndex3D> quads;
  std::vector<local_shell_t> shell;
  // Reserve some memory to reduce the number of reallocs.
  // quads and shell also hold ghost cells, so this does not prevent
  // reallocs
  quads.reserve(dd.p4est->local_num_quadrants);
  shell.reserve(dd.p4est->local_num_quadrants);

  // Loop all local cells to gather information for those
  for (int i = 0; i < dd.p4est->local_num_quadrants; ++i) {
    p4est_quadrant_t *q = p4est_mesh_get_quadrant(dd.p4est, p4est_mesh, i);
    quad_data_t *data = (quad_data_t *)(q->p.user_data);
    double xyz[3];
    p4est_qcoord_to_vertex(dd.p4est_conn, p4est_mesh->quad_to_tree[i], q->x,
                           q->y, q->z, xyz);
    uint64_t ql =
        1
        << p4est_tree_array_index(dd.p4est->trees, p4est_mesh->quad_to_tree[i])->maxlevel;
    uint64_t x = xyz[0] * ql;
    uint64_t y = xyz[1] * ql;
    uint64_t z = xyz[2] * ql;
    // This is a simple but easy unique index, that also works for cells outside
    // box_l (by one cell)
    quads.push_back(idx3d(x + 1, y + 1, z + 1));
    local_shell_t ls;
    ls.idx = i;
    ls.rank = this_node;
    ls.shell = CellType::inner;
    ls.boundary = 0;
    ls.coord[0] = x;
    ls.coord[1] = y;
    ls.coord[2] = z;
    ls.p_cnt = 0;
    // Gather all inforamtion about neighboring cells
    for (int n = 0; n < 26; ++n) {
      ls.neighbor[n] = -1;
      sc_array_t *ne, *ni;
      ne = sc_array_new(sizeof(int));
      ni = sc_array_new(sizeof(int));
      p4est_mesh_get_neighbors(dd.p4est, p4est_ghost, p4est_mesh, i, -1, n, 0,
                               NULL, ne, ni);
      if (ni->elem_count > 1)
        printf("%i %i %li strange stuff\n", i, n, ni->elem_count);
      if (ni->elem_count > 0) {
        data->ishell[n] = *((int *)sc_array_index_int(ni, 0));
        if (*((int *)sc_array_index_int(ne, 0)) >= 0) // Local cell
          data->rshell[n] = this_node;
        else { // in ghost layer
          data->rshell[n] = p4est_mesh->ghost_to_proc[data->ishell[n]];
        }
      }
      sc_array_destroy(ne);
      sc_array_destroy(ni);
    }
    shell.push_back(std::move(ls));
  }


  // compute ghost, mirror and boundary information
  // here the ghost layer around the local domain is computed
  for (int i = 0; i < dd.p4est->local_num_quadrants; ++i) {
    p4est_quadrant_t *q = p4est_mesh_get_quadrant(dd.p4est, p4est_mesh, i);
    quad_data_t *data = (quad_data_t *)(q->p.user_data);
    double xyz[3];
    p4est_qcoord_to_vertex(dd.p4est_conn, p4est_mesh->quad_to_tree[i], q->x,
                           q->y, q->z, xyz);
    uint64_t ql = 1 << p4est_tree_array_index(dd.p4est->trees,
                                              p4est_mesh->quad_to_tree[i])
                                                  ->maxlevel;
    uint64_t x = xyz[0] * ql;
    uint64_t y = xyz[1] * ql;
    uint64_t z = xyz[2] * ql;

    // Loop all 27 cells in the fullshell
    for (uint64_t zi = 0; zi <= 2; zi++)
      for (uint64_t yi = 0; yi <= 2; yi++)
        for (uint64_t xi = 0; xi <= 2; xi++) {
          if (xi == 1 && yi == 1 && zi == 1)
            continue;
          // Check if this node has already been processed using the unique
          // index
          FastIndex3D qidx = idx3d(x + xi, y + yi, z + zi);
          auto it = std::find(std::begin(quads), std::end(quads), qidx);

          if (it == std::end(quads)) { // Cell has not been processed yet
            quads.push_back(qidx);   // Add it to list
            local_shell_t ls;
            // Copy corresponding information from p4est internal struct
            ls.idx = data->ishell[neighbor_lut[zi][yi][xi]];
            ls.rank = data->rshell[neighbor_lut[zi][yi][xi]];
            ls.shell = CellType::ghost; // This is a ghost cell, since all locals have been
                          // added before
            ls.boundary = dd_p4est_boundary_flags(x + xi, y + yi, z + zi);

            std::fill(std::begin(ls.neighbor), std::end(ls.neighbor), -1);

            shell[i].boundary |= dd_p4est_boundary_flags_for_neighbor(xi, yi, zi);
            // Link the new cell to a local cell
            shell[i].neighbor[neighbor_lut[zi][yi][xi]] = shell.size();
            ls.coord[0] = int(x + xi) - 1;
            ls.coord[1] = int(y + yi) - 1;
            ls.coord[2] = int(z + zi) - 1;
            ls.p_cnt = 0;
            if (ls.rank == this_node)
              ls.p_cnt = 1;
            for (uint64_t l = num_local_cells; l < shell.size(); ++l) {
              if (shell[l].idx == ls.idx && shell[l].rank == ls.rank) {
                if (shell[l].boundary < ls.boundary)
                  ls.p_cnt += 1;
                else
                  shell[l].p_cnt += 1;
              }
            }
            shell.push_back(std::move(ls)); // add the new ghost cell to all cells
            shell[i].shell = CellType::boundary;  // The cell for which this one was added is at
                                 // the domain bound
          } else {               // Cell already exists in list
            auto pos = std::distance(std::begin(quads), it);
            if (shell[pos].shell == CellType::ghost) { // is it a ghost cell, then update the boundary info
              // of the current local cell, since they are neighbors
              shell[i].shell = CellType::boundary; // this local cell is at domain boundary
              // Update boundary info
              shell[i].boundary |= dd_p4est_boundary_flags_for_neighbor(xi, yi, zi);
            }
            // Link it as neighbor
            shell[i].neighbor[neighbor_lut[zi][yi][xi]] = pos;
          }
        }
  }

  // Copy the generated data to globals
  num_cells = (size_t)quads.size();
  num_local_cells = (size_t)dd.p4est->local_num_quadrants;
  num_ghost_cells = num_cells - (size_t)dd.p4est->local_num_quadrants;

  dd.p4est_shell = std::move(shell);

  CELL_TRACE(printf("%d : %ld, %ld, %ld\n", this_node, num_cells,
                    num_local_cells, num_ghost_cells));

#ifndef P4EST_NOCHANGE
  // allocate memory
  realloc_cells(num_cells);
  realloc_cellplist(&local_cells, local_cells.n = num_local_cells);
  realloc_cellplist(&ghost_cells, ghost_cells.n = num_ghost_cells);
#endif

  //dd_p4est_write_vtk();

  CELL_TRACE(printf("%d: %.3f %.3fx%.3fx%.3f %.3fx%.3fx%.3f\n", this_node,
                    max_range, box_l[0], box_l[1], box_l[2], dd.cell_size[0],
                    dd.cell_size[1], dd.cell_size[2]));
}
//--------------------------------------------------------------------------------------------------

// Compute communication partners and the cells that need to be comunicated
void dd_p4est_comm() {
  CALL_TRACE();

  // List of cell idx marked for send/recv for each process
  std::vector<std::vector<int>> send_idx(n_nodes), recv_idx(n_nodes);
  // List of all directions for those communicators encoded in a bitmask
  std::vector<std::vector<uint64_t>> send_tag(n_nodes);
  std::vector<std::vector<uint64_t>> recv_tag(n_nodes);
  // Or-Sum (Union) over the lists above
  std::vector<uint64_t> send_cnt_tag(n_nodes, 0);
  std::vector<uint64_t> recv_cnt_tag(n_nodes, 0);
  // Number of cells for each communication (rank and direction)
  // 64 different direction (64 = 2^6, where 6 = |{x_l, x_r, y_l, y_r, z_l,
  // z_r}|)
  std::vector<std::array<int, 64>> send_cnt(n_nodes);
  std::vector<std::array<int, 64>> recv_cnt(n_nodes);

  for (auto &a : send_cnt)
    a.fill(0);
  for (auto &a : recv_cnt)
    a.fill(0);

  // Total number of send and recv
  int num_send = 0;
  int num_recv = 0;

  // Is 1 for a process if there is any communication, 0 if none
  std::vector<int8_t> num_send_flag(n_nodes, 0);
  std::vector<int8_t> num_recv_flag(n_nodes, 0);

  // Prepare all lists
  num_comm_proc = 0;
  comm_proc.resize(n_nodes, -1);

  // create send and receive list
  // char fname[100];
  // sprintf(fname,"cells_conn_%i.list",this_node);
  // FILE* h = fopen(fname,"w");
  // Loop all cells
  for (int i = 0; i < num_cells; ++i) {
    // is ghost cell that is linked to a process? -> add to recv list
    if (dd.p4est_shell[i].rank >= 0 && dd.p4est_shell[i].shell == CellType::ghost) {
      int irank = dd.p4est_shell[i].rank;
      int pos = 0;
      // find position to add new element (keep order)
      while (pos < recv_idx[irank].size() &&
             dd.p4est_shell[recv_idx[irank][pos]].idx <= dd.p4est_shell[i].idx)
        pos++;

      if (pos >= recv_idx[irank].size()) { // Add to end of vector
        recv_idx[irank].push_back(i);
        recv_tag[irank].push_back(1L << dd.p4est_shell[i].boundary);
        // insert if this cell has not been added yet
      } else if (dd.p4est_shell[recv_idx[irank][pos]].idx !=
                 dd.p4est_shell[i].idx) {
        recv_idx[irank].insert(recv_idx[irank].begin() + pos, i);
        recv_tag[irank].insert(recv_tag[irank].begin() + pos,
                               1L << dd.p4est_shell[i].boundary);
        // update diraction info for communication if already added but for
        // other direction
      } else {
        recv_tag[irank][pos] |= 1L << dd.p4est_shell[i].boundary;
      }
      // count what happend above
      recv_cnt[irank][dd.p4est_shell[i].boundary] += 1;
      if ((recv_cnt_tag[irank] & (1L << dd.p4est_shell[i].boundary)) == 0) {
        ++num_recv;
        recv_cnt_tag[irank] |= 1L << dd.p4est_shell[i].boundary;
      }
      // recv_cnt[dd.p4est_shell[i].rank].insert(recv_cnt[dd.p4est_shell[i].rank].begin()
      // + pos, i); //dd.p4est_shell[i].idx);
      if (num_recv_flag[irank] == 0) {
        //++num_recv;
        comm_proc[irank] = num_comm_proc;
        num_comm_proc += 1;
        num_recv_flag[irank] = 1;
      }
    }
    // is mirror cell (at domain boundary)? -> add to send list
    if (dd.p4est_shell[i].shell == CellType::boundary) {
      // for (int n=0;n<n_nodes;++n) comm_cnt[n] = 0;
      // loop fullshell
      for (int n = 0; n < 26; ++n) {
        int nidx = dd.p4est_shell[i].neighbor[n];
        int nrank = dd.p4est_shell[nidx].rank;
        if (nidx < 0 || nrank < 0)
          continue; // invalid neighbor
        if (dd.p4est_shell[nidx].shell != CellType::ghost)
          continue; // no need to send to local cell
        // check if this is the first time to add this mirror cell
        if (!send_tag[nrank].empty() &&
            send_idx[nrank].back() == i) { // already added
          if ((send_tag[nrank].back() & (1L << dd.p4est_shell[nidx].boundary)))
            continue;
          // update direction info for this communication
          send_tag[nrank].back() |= (1L << dd.p4est_shell[nidx].boundary);
        } else { // not added yet -> do so
          send_idx[nrank].push_back(i);
          send_tag[nrank].push_back(1L << dd.p4est_shell[nidx].boundary);
        }
        // count what happend
        send_cnt[nrank][dd.p4est_shell[nidx].boundary] += 1;
        if ((send_cnt_tag[nrank] & (1L << dd.p4est_shell[nidx].boundary)) ==
            0) {
          ++num_send;
          send_cnt_tag[nrank] |= 1L << dd.p4est_shell[nidx].boundary;
        }
        if (num_send_flag[nrank] == 0) {
          //++num_send;
          num_send_flag[nrank] = 1;
        }
      }
    }
    // fprintf(h,"%i %i:%li (%i) %i %i [
    // ",i,dd.p4est_shell[i].rank,dd.p4est_shell[i].idx,dd.p4est_shell[i].p_cnt,
    //  dd.p4est_shell[i].shell,dd.p4est_shell[i].boundary);
    // for (int n=0;n<26;++n) fprintf(h,"%i ",dd.p4est_shell[i].neighbor[n]);
    // fprintf(h,"]\n");
  }
  // fclose(h);
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
      fprintf(h,"%i:%li
  0x%016lx\n",n,p4est_shell[recv_idx[n][i]].idx,recv_tag[n][i]);
  fclose(h);*/

  // prepare communicator
  CELL_TRACE(fprintf(stdout, "%i : proc %i send %i, recv %i\n", this_node,
                     num_comm_proc, num_send, num_recv));
  num_comm_recv = num_recv;
  num_comm_send = num_send;
  comm_recv.resize(num_recv);
  comm_send.resize(num_send);
  comm_rank.resize(num_comm_proc);

  // Parse all bitmasks and fill the actual lists
  for (int n = 0, s_cnt = 0, r_cnt = 0; n < n_nodes; ++n) {
    if (comm_proc[n] >= 0)
      comm_rank[comm_proc[n]] = n;
    for (int i = 0; i < 64; ++i) {
      if (num_recv_flag[n] && (recv_cnt_tag[n] & 1L << i)) {
        comm_recv[r_cnt].cnt = recv_cnt[n][i];
        comm_recv[r_cnt].rank = n;
        comm_recv[r_cnt].dir = i;
        comm_recv[r_cnt].idx.resize(recv_cnt[n][i]);
        for (int j = 0, c = 0; j < recv_idx[n].size(); ++j)
          if ((recv_tag[n][j] & (1L << i)))
            comm_recv[r_cnt].idx[c++] = recv_idx[n][j];
        ++r_cnt;
      }
      if (num_send_flag[n] && (send_cnt_tag[n] & 1L << i)) {
        comm_send[s_cnt].cnt = send_cnt[n][i];
        comm_send[s_cnt].rank = n;
        comm_send[s_cnt].dir = i;
        comm_send[s_cnt].idx.resize(send_cnt[n][i]);
        for (int j = 0, c = 0; j < send_idx[n].size(); ++j)
          if ((send_tag[n][j] & 1L << i))
            comm_send[s_cnt].idx[c++] = send_idx[n][j];
        ++s_cnt;
      }
    }
  }

  /*sprintf(fname,"send_%i.list",this_node);
  h = fopen(fname,"w");
  for (int n=0;n<num_comm_send;++n)
    for (int i=0;i<comm_send[n].cnt;++i)
      fprintf(h,"%i:%i
  %i\n",comm_send[n].rank,comm_send[n].idx[i],comm_send[n].dir);
  fclose(h);
  sprintf(fname,"recv_%i.list",this_node);
  h = fopen(fname,"w");
  for (int n=0;n<num_comm_recv;++n)
    for (int i=0;i<comm_recv[n].cnt;++i)
      fprintf(h,"%i:%i
  %i\n",comm_recv[n].rank,comm_recv[n].idx[i],comm_recv[n].dir);
  fclose(h);*/
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_prepare_comm(GhostCommunicator *comm, int data_part) {
  CALL_TRACE();
  prepare_comm(comm, data_part, num_comm_send + num_comm_recv, true);
  int cnt = 0;
  for (int i = 0; i < num_comm_send; ++i) {
    comm->comm[cnt].type = GHOST_SEND;
    comm->comm[cnt].node = comm_send[i].rank;
    // The tag distinguishes communications to the same rank
    comm->comm[cnt].tag = comm_send[i].dir;
    comm->comm[cnt].part_lists = (ParticleList **)Utils::malloc(
        comm_send[i].cnt * sizeof(ParticleList *));
    comm->comm[cnt].n_part_lists = comm_send[i].cnt;
    for (int n = 0; n < comm_send[i].cnt; ++n)
      comm->comm[cnt].part_lists[n] = &cells[comm_send[i].idx[n]];
    if ((data_part & GHOSTTRANS_POSSHFTD)) {
      // Set shift according to communication direction
      if ((comm_send[i].dir & 1))
        comm->comm[cnt].shift[0] = box_l[0];
      if ((comm_send[i].dir & 2))
        comm->comm[cnt].shift[0] = -box_l[0];
      if ((comm_send[i].dir & 4))
        comm->comm[cnt].shift[1] = box_l[1];
      if ((comm_send[i].dir & 8))
        comm->comm[cnt].shift[1] = -box_l[1];
      if ((comm_send[i].dir & 16))
        comm->comm[cnt].shift[2] = box_l[2];
      if ((comm_send[i].dir & 32))
        comm->comm[cnt].shift[2] = -box_l[2];
    }
    ++cnt;
  }
  for (int i = 0; i < num_comm_recv; ++i) {
    comm->comm[cnt].type = GHOST_RECV;
    comm->comm[cnt].node = comm_recv[i].rank;
    comm->comm[cnt].tag = comm_recv[i].dir; // The tag is the same as above
    // But invert x, y and z direction now. Direction from where the data came
    if ((comm_recv[i].dir & 3))
      comm->comm[cnt].tag ^= 3;
    if ((comm_recv[i].dir & 12))
      comm->comm[cnt].tag ^= 12;
    if ((comm_recv[i].dir & 48))
      comm->comm[cnt].tag ^= 48;
    comm->comm[cnt].part_lists = (ParticleList **)Utils::malloc(
        comm_recv[i].cnt * sizeof(ParticleList *));
    comm->comm[cnt].n_part_lists = comm_recv[i].cnt;
    for (int n = 0; n < comm_recv[i].cnt; ++n)
      comm->comm[cnt].part_lists[n] = &cells[comm_recv[i].idx[n]];
    ++cnt;
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_mark_cells() {
  CALL_TRACE();
// Take the memory map as they are. First local cells (along Morton curve).
// This also means cells has the same ordering as p4est_shell
#ifndef P4EST_NOCHANGE
  for (int c = 0; c < num_local_cells; ++c)
    local_cells.cell[c] = &cells[c];
  // Ghost cells come after local cells
  for (int c = 0; c < num_ghost_cells; ++c)
    ghost_cells.cell[c] = &cells[num_local_cells + c];
#endif
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_update_comm_w_boxl(GhostCommunicator *comm) {
  CALL_TRACE();
  int cnt = 0;
  for (int i = 0; i < num_comm_send; ++i) {
    // Reset shift according to communication direction
    comm->comm[cnt].shift[0] = comm->comm[cnt].shift[1] =
        comm->comm[cnt].shift[2] = 0.0;
    if ((comm_send[i].dir & 1))
      comm->comm[cnt].shift[0] = box_l[0];
    if ((comm_send[i].dir & 2))
      comm->comm[cnt].shift[0] = -box_l[0];
    if ((comm_send[i].dir & 4))
      comm->comm[cnt].shift[1] = box_l[1];
    if ((comm_send[i].dir & 8))
      comm->comm[cnt].shift[1] = -box_l[1];
    if ((comm_send[i].dir & 16))
      comm->comm[cnt].shift[2] = box_l[2];
    if ((comm_send[i].dir & 32))
      comm->comm[cnt].shift[2] = -box_l[2];
    ++cnt;
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_init_cell_interaction() {
#ifndef P4EST_NOCHANGE
  dd.cell_inter = (IA_Neighbor_List *)Utils::realloc(
      dd.cell_inter, local_cells.n * sizeof(IA_Neighbor_List));
  for (int i = 0; i < local_cells.n; i++) {
    dd.cell_inter[i].nList = NULL;
    dd.cell_inter[i].n_neighbors = 0;
  }

  for (int i = 0; i < num_local_cells; ++i) {
    dd.cell_inter[i].nList = (IA_Neighbor *)Utils::realloc(
        dd.cell_inter[i].nList, CELLS_MAX_NEIGHBORS * sizeof(IA_Neighbor));

    // Copy info of the local cell itself
    dd.cell_inter[i].nList[0].cell_ind = i;
    dd.cell_inter[i].nList[0].pList = &cells[i];
    init_pairList(&dd.cell_inter[i].nList[0].vList);

    // Copy all other cells in half-shell
    for (int n = 1; n < CELLS_MAX_NEIGHBORS; ++n) {
      dd.cell_inter[i].nList[n].cell_ind =
          dd.p4est_shell[i].neighbor[half_neighbor_idx[n]];
      dd.cell_inter[i].nList[n].pList =
          &cells[dd.p4est_shell[i].neighbor[half_neighbor_idx[n]]];
      init_pairList(&dd.cell_inter[i].nList[n].vList);
    }

    dd.cell_inter[i].n_neighbors = CELLS_MAX_NEIGHBORS;
  }
#endif
}
//--------------------------------------------------------------------------------------------------

Cell *dd_p4est_save_position_to_cell(double pos[3]) {
  int64_t i = p4est_utils_pos_quad_ext(forest_order::short_range, pos);

  // return the index
  if (i >= 0 && i < num_local_cells)
    return &cells[i];
  return NULL;
}
//--------------------------------------------------------------------------------------------------
Cell *dd_p4est_position_to_cell(double pos[3]) {
  CALL_TRACE();

  // Does the same as dd_p4est_save_position_to_cell but does not extend the
  // local domain
  // by the error bounds
  int i = p4est_utils_pos_qid_local(forest_order::short_range, pos);
  P4EST_ASSERT(0 <= i && i < dd.p4est->local_num_quadrants);

  if (i < num_local_cells)
    return &cells[i];
  return NULL;
}
//--------------------------------------------------------------------------------------------------
// Checks all particles and resorts them to local cells or sendbuffers
// Note: Particle that stay local and are moved to a cell with higher index are
// touched twice.
// This is not the most efficient way! It would be better to first remember the
// cell-particle
// index pair of those particles in a vector and move them at the end.
void dd_p4est_fill_sendbuf(ParticleList *sendbuf,
                           std::vector<int> *sendbuf_dyn) {
  double cell_lc[3], cell_hc[3];
  // Loop over all cells and particles
  for (int i = 0; i < num_local_cells; ++i) {
    Cell *cell = local_cells.cell[i];
    local_shell_t *shell = &dd.p4est_shell[i];

    for (int d = 0; d < 3; ++d) {
      cell_lc[d] = dd.cell_size[d] * (double)shell->coord[d];
      cell_hc[d] = cell_lc[d] + dd.cell_size[d];
      if ((shell->boundary & (1 << (2 * d))))
        cell_lc[d] -= 0.5 * ROUND_ERROR_PREC * box_l[d];
      if ((shell->boundary & (2 << (2 * d))))
        cell_hc[d] += 0.5 * ROUND_ERROR_PREC * box_l[d];
    }

    for (int p = 0; p < cell->n; ++p) {
      Particle *part = &cell->part[p];
      int x, y, z;

      // Check if particle has left the cell. (The local domain is extenden by
      // half round error)
      if (part->r.p[0] < cell_lc[0])
        x = 0;
      else if (part->r.p[0] >= cell_hc[0])
        x = 2;
      else
        x = 1;
      if (part->r.p[1] < cell_lc[1])
        y = 0;
      else if (part->r.p[1] >= cell_hc[1])
        y = 2;
      else
        y = 1;
      if (part->r.p[2] < cell_lc[2])
        z = 0;
      else if (part->r.p[2] >= cell_hc[2])
        z = 2;
      else
        z = 1;

      int nidx = neighbor_lut[z][y][x];
      if (nidx != -1) { // Particle p outside of cell i
        // recalculate neighboring cell to prevent rounding errors
        // If particle left local domain, check for correct ghost cell, thus
        // without ROUND_ERROR_PREC
        for (int d = 0; d < 3; ++d) {
          cell_lc[d] = dd.cell_size[d] * (double)shell->coord[d];
          cell_hc[d] = cell_lc[d] + dd.cell_size[d];
        }
        if (part->r.p[0] < cell_lc[0])
          x = 0;
        else if (part->r.p[0] >= cell_hc[0])
          x = 2;
        else
          x = 1;
        if (part->r.p[1] < cell_lc[1])
          y = 0;
        else if (part->r.p[1] >= cell_hc[1])
          y = 2;
        else
          y = 1;
        if (part->r.p[2] < cell_lc[2])
          z = 0;
        else if (part->r.p[2] >= cell_hc[2])
          z = 2;
        else
          z = 1;

        nidx = neighbor_lut[z][y][x];
        // get neighbor cell
        nidx = shell->neighbor[nidx];
        if (nidx >= num_local_cells) { // Remote Cell (0:num_local_cells-1) ->
                                       // local, other: ghost
          if (dd.p4est_shell[nidx].rank >=
              0) { // This ghost cell is linked to a process
            CELL_TRACE(fprintf(stderr, "%d: dd_ex_and_sort_p: send part %d\n",
                               this_node, part->p.identity));

            if (dd.p4est_shell[nidx].rank !=
                this_node) { // It is a remote process
              // copy data to sendbuf according to rank
              int li = comm_proc[dd.p4est_shell[nidx].rank];
              sendbuf_dyn[li].insert(sendbuf_dyn[li].end(), part->bl.e,
                                     part->bl.e + part->bl.n);
#ifdef EXCLUSIONS
              sendbuf_dyn[li].insert(sendbuf_dyn[li].end(), part->el.e,
                                     part->el.e + part->el.n);
#endif
              int pid = part->p.identity;
              // fold_position(part->r.p, part->l.i);
              move_indexed_particle(&sendbuf[li], cell, p);
              local_particles[pid] = NULL;
              if (p < cell->n)
                p -= 1;
            } else { // particle stays local, but since it went to a ghost it
                     // has to be folded
              fold_position(part->r.p, part->l.i);
              move_indexed_particle(&cells[dd.p4est_shell[nidx].idx], cell, p);
              if (p < cell->n)
                p -= 1;
            }
          } else { // Particle left global domain and is not tracked by any
                   // process anymore
            runtimeErrorMsg()
                << "particle " << p << " on process " << this_node << " is OB";
            fprintf(stderr, "%i : part %i cell %i is OB [%lf %lf %lf]\n",
                    this_node, i, p, part->r.p[0], part->r.p[1], part->r.p[2]);
          }
        } else { // Local Cell, just move the partilce
          move_indexed_particle(&cells[nidx], cell, p);
          if (p < cell->n)
            p -= 1;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------------------------------
static int dd_async_exchange_insert_particles(ParticleList *recvbuf,
                                              int global_flag, int from) {
  // add all particle in a recvbuf to the local storage
  int dynsiz = 0;

  update_local_particles(recvbuf);

  for (int p = 0; p < recvbuf->n; ++p) {
    double op[3] = {recvbuf->part[p].r.p[0], recvbuf->part[p].r.p[1],
                    recvbuf->part[p].r.p[2]};
    fold_position(recvbuf->part[p].r.p, recvbuf->part[p].l.i);

    dynsiz += recvbuf->part[p].bl.n;
#ifdef EXCLUSIONS
    dynsiz += recvbuf->part[p].el.n;
#endif
    //}
    // Fold direction of dd_append_particles unused.

    // for (int p=0;p<recvbuf->n;++p) {
    Cell *target = dd_p4est_save_position_to_cell(recvbuf->part[p].r.p);
    if (target) {
      append_indexed_particle(target, &recvbuf->part[p]);
    } else {
      fprintf(stderr, "proc %i received remote particle p%i out of domain, "
                      "global %i from proc %i\n\t%lfx%lfx%lf, glob morton idx "
                      "%i, pos2proc %i\n\told pos %lfx%lfx%lf\n",
              this_node, recvbuf->part[p].p.identity, global_flag, from,
              recvbuf->part[p].r.p[0], recvbuf->part[p].r.p[1],
              recvbuf->part[p].r.p[2],
              p4est_utils_pos_qid_local(forest_order::short_range,
                                        recvbuf->part[p].r.p),
              dd_p4est_pos_to_proc(recvbuf->part[p].r.p), op[0], op[1], op[2]);
      errexit();
    }
  }

  return dynsiz;
}
//--------------------------------------------------------------------------------------------------
static void dd_async_exchange_insert_dyndata(ParticleList *recvbuf,
                                             std::vector<int> &dynrecv) {
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
      memcpy(p->el.e, &dynrecv[read], p->el.n * sizeof(int));
      read += p->el.n;
    } else {
      p->el.e = NULL;
    }
#endif
  }
}
//--------------------------------------------------------------------------------------------------

void dd_p4est_exchange_and_sort_particles() {
  // Prepare all send and recv buffers to all neighboring processes
  std::vector<MPI_Request> sreq(3 * num_comm_proc, MPI_REQUEST_NULL);
  std::vector<MPI_Request> rreq(num_comm_proc, MPI_REQUEST_NULL);
  std::vector<int> nrecvpart(num_comm_proc, 0);

  std::vector<ParticleList> sendbuf(num_comm_proc);
  std::vector<ParticleList> recvbuf(num_comm_proc);
  std::vector<std::vector<int>> sendbuf_dyn(num_comm_proc);
  std::vector<std::vector<int>> recvbuf_dyn(num_comm_proc);

  for (int i = 0; i < num_comm_proc; ++i) {
    init_particlelist(&sendbuf[i]);
    init_particlelist(&recvbuf[i]);
    // Invoke the recv thread for number of particles from all neighbouring
    // nodes
    MPI_Irecv(&nrecvpart[i], 1, MPI_INT, comm_rank[i], 0, comm_cart, &rreq[i]);
  }

  // Fill the send buffers with particles that leave the local domain
  dd_p4est_fill_sendbuf(sendbuf.data(), sendbuf_dyn.data());

  // send number of particles, particles, and particle data
  for (int i = 0; i < num_comm_proc; ++i) {
    MPI_Isend(&sendbuf[i].n, 1, MPI_INT, comm_rank[i], 0, comm_cart, &sreq[i]);
    MPI_Isend(sendbuf[i].part, sendbuf[i].n * sizeof(Particle), MPI_BYTE,
              comm_rank[i], 1, comm_cart, &sreq[i + num_comm_proc]);
    if (sendbuf_dyn[i].size() > 0) {
      MPI_Isend(sendbuf_dyn[i].data(), sendbuf_dyn[i].size(), MPI_INT,
                comm_rank[i], 2, comm_cart, &sreq[i + 2 * num_comm_proc]);
    }
  }

  // Receive all data
  MPI_Status status;
  std::vector<int> recvs(num_comm_proc, 0); // number of recv for each process
  int recvidx, source;
  while (true) {
    MPI_Waitany(num_comm_proc, rreq.data(), &recvidx, &status);
    if (recvidx == MPI_UNDEFINED)
      break;

    source = status.MPI_SOURCE;

    int cnt;
    MPI_Get_count(&status, MPI_BYTE, &cnt);

    // this is the first recv for the process. this means, that the Irecv
    // fornumber of particles
    // has finished
    if (recvs[recvidx] == 0) {
      // alloc recv buffer with number of particles
      realloc_particlelist(&recvbuf[recvidx], nrecvpart[recvidx]);
      // fill it
      MPI_Irecv(recvbuf[recvidx].part, nrecvpart[recvidx] * sizeof(Particle),
                MPI_BYTE, source, 1, comm_cart, &rreq[recvidx]);
    } else if (recvs[recvidx] == 1) { // filling the recv buffer has finished
      // Particles received
      recvbuf[recvidx].n = nrecvpart[recvidx];
      // Add new particles to local storage
      int dyndatasiz =
          dd_async_exchange_insert_particles(&recvbuf[recvidx], 0, source);
      if (dyndatasiz >
          0) { // If there is dynamic data, invoke recv thread for that as well
        recvbuf_dyn[recvidx].resize(dyndatasiz);
        MPI_Irecv(recvbuf_dyn[recvidx].data(), dyndatasiz, MPI_INT, source,
                  /*tag*/ 2, comm_cart, &rreq[recvidx]);
      }
    } else { // recv for dynamic data has finished
      dd_async_exchange_insert_dyndata(&recvbuf[recvidx], recvbuf_dyn[recvidx]);
    }
    recvs[recvidx]++;
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
  }

#ifdef ADDITIONAL_CHECKS
  check_particle_consistency();
#endif
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_global_exchange_part(ParticleList *pl) {
  // Prepare send/recv buffers to ALL processes
  std::vector<ParticleList> sendbuf(n_nodes), recvbuf(n_nodes);
  std::vector<std::vector<int>> sendbuf_dyn(n_nodes), recvbuf_dyn(n_nodes);
  std::vector<MPI_Request> sreq(3 * n_nodes, MPI_REQUEST_NULL);
  std::vector<MPI_Request> rreq(n_nodes, MPI_REQUEST_NULL);
  std::vector<int> nrecvpart(n_nodes, 0);

  for (int i = 0; i < n_nodes; ++i) {
    init_particlelist(&sendbuf[i]);
    init_particlelist(&recvbuf[i]);

    MPI_Irecv(&nrecvpart[i], 1, MPI_INT, i, 0, comm_cart, &rreq[i]);
  }

  // If pl == NULL do nothing, since there are no particles and cells on this
  // node
  if (pl) {
    // Find the correct node for each particle in pl
    for (int p = 0; p < pl->n; p++) {
      Particle *part = &pl->part[p];
      // fold_position(part->r.p, part->l.i);
      int rank = dd_p4est_pos_to_proc(part->r.p);
      if (rank != this_node) { // It is actually a remote particle -> copy all
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
    MPI_Isend(&sendbuf[i].n, 1, MPI_INT, i, 0, comm_cart, &sreq[i]);
    MPI_Isend(sendbuf[i].part, sendbuf[i].n * sizeof(Particle), MPI_BYTE, i, 0,
              comm_cart, &sreq[i + n_nodes]);
    if (sendbuf_dyn[i].size() > 0)
      MPI_Isend(sendbuf_dyn[i].data(), sendbuf_dyn[i].size(), MPI_INT, i, 0,
                comm_cart, &sreq[i + 2 * n_nodes]);
  }

  // Receive all data. The async communication scheme is the same as in
  // exchange_and_sort_particles
  MPI_Status status;
  std::vector<int> recvs(n_nodes, 0);
  int recvidx, tag, source;
  while (true) {
    MPI_Waitany(n_nodes, rreq.data(), &recvidx, &status);
    if (recvidx == MPI_UNDEFINED)
      break;

    source = status.MPI_SOURCE;
    tag = status.MPI_TAG;

    if (recvs[recvidx] == 0) {
      // Size received
      realloc_particlelist(&recvbuf[recvidx], nrecvpart[recvidx]);
      MPI_Irecv(recvbuf[recvidx].part, nrecvpart[recvidx] * sizeof(Particle),
                MPI_BYTE, source, tag, comm_cart, &rreq[recvidx]);
    } else if (recvs[recvidx] == 1) {
      // Particles received
      recvbuf[recvidx].n = nrecvpart[recvidx];
      int dyndatasiz =
          dd_async_exchange_insert_particles(&recvbuf[recvidx], 1, source);
      if (dyndatasiz > 0) {
        recvbuf_dyn[recvidx].resize(dyndatasiz);
        MPI_Irecv(recvbuf_dyn[recvidx].data(), dyndatasiz, MPI_INT, source, tag,
                  comm_cart, &rreq[recvidx]);
      }
    } else {
      dd_async_exchange_insert_dyndata(&recvbuf[recvidx], recvbuf_dyn[recvidx]);
    }
    recvs[recvidx]++;
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
// Find the process that handles the position
int dd_p4est_pos_to_proc(double pos[3]) {
  return p4est_utils_pos_to_proc(forest_order::short_range, pos);
}
//--------------------------------------------------------------------------------------------------
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
      fprintf(h, "\t\t\t\t\t%le %le %le\n", part[p].r.p[0], part[p].r.p[1],
              part[p].r.p[2]);
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
void dd_p4est_write_vtk() {
  // write cells to VTK with a given streching
  p4est_vtk_write_file(dd.p4est, NULL, P4EST_STRING "_dd");

  /*char fname[100];
  sprintf(fname,"cells_conn_%i.list",this_node);
  FILE* h = fopen(fname,"w");
  for (int i=0;i<num_cells;++i) {
    fprintf(h,"%i %i:%li (%i) %i %i [ ",i, p4est_shell[i].rank,
            dd.p4est_shell[i].idx, dd.p4est_shell[i].p_cnt,
            dd.p4est_shell[i].shell, dd.p4est_shell[i].boundary);
    for (int n = 0; n < 26; ++n)
      fprintf(h,"%i ", dd.p4est_shell[i].neighbor[n]);
    fprintf(h,"]\n");
  }
  fclose(h);*/
}
//--------------------------------------------------------------------------------------------------
#endif // DD_P4EST

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributors
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

#include "pair_mace_kokkos.h"

// TODO: do i need all these?
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMACEKokkos<DeviceType>::PairMACEKokkos(LAMMPS *lmp) : PairMACE(lmp)
{
  no_virial_fdotr_compute = 1;

  // TODO: from lj_cut ... possibly delete
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  //datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMACEKokkos<DeviceType>::~PairMACEKokkos()
{
  // TODO: from lj_cut, possibly delete
  if (copymode) return;

  // TODO: from lj_cut, possibly delete
  //if (allocated) {
  //  memoryKK->destroy_kokkos(k_eatom,eatom);
  //  memoryKK->destroy_kokkos(k_vatom,vatom);
  //  memoryKK->destroy_kokkos(k_cutsq,cutsq);
  //}
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMACEKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag,0);

  if (eflag_atom || vflag_atom) {
    error->all(FLERR, "ERROR: kokkos eflag_atom not implemented.");
  }

//  if (eflag_atom) {
//    memoryKK->destroy_kokkos(k_eatom,eatom);
//    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
//    d_eatom = k_eatom.view<DeviceType>();
//  }
//  if (vflag_atom) {
//    memoryKK->destroy_kokkos(k_vatom,vatom);
//    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
//    d_vatom = k_vatom.view<DeviceType>();
//  }

  atomKK->sync(execution_space,datamask_read);
  //k_cutsq.template sync<DeviceType>();
  //k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  //newton_pair = force->newton_pair;
  //special_lj[0] = force->special_lj[0];
  //special_lj[1] = force->special_lj[1];
  //special_lj[2] = force->special_lj[2];
  //special_lj[3] = force->special_lj[3];

  // loop over neighbors of my atoms

  //copymode = 1;

  //EV_FLOAT ev = pair_compute<PairLJCutKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);

  //if (eflag_global) eng_vdwl += ev.evdwl;
  //if (vflag_global) {
  //  virial[0] += ev.v[0];
  //  virial[1] += ev.v[1];
  //  virial[2] += ev.v[2];
  //  virial[3] += ev.v[3];
  //  virial[4] += ev.v[4];
  //  virial[5] += ev.v[5];
  //}

  //if (eflag_atom) {
  //  k_eatom.template modify<DeviceType>();
  //  k_eatom.template sync<LMPHostType>();
  //}

  //if (vflag_atom) {
  //  k_vatom.template modify<DeviceType>();
  //  k_vatom.template sync<LMPHostType>();
  //}

  copymode = 0;
  

//  ev_init(eflag, vflag);
//
//  if (atom->nlocal != list->inum) error->all(FLERR, "ERROR: nlocal != inum.");
//  if (domain_decomposition) {
//    if (atom->nghost != list->gnum) error->all(FLERR, "ERROR: nghost != gnum.");
//  }
//
//  // ----- positions -----
//  int n_nodes;
//  if (domain_decomposition) {
//    n_nodes = atom->nlocal + atom->nghost;
//  } else {
//    // normally, ghost atoms are included in the graph as independent
//    // nodes, as required when the local domain does not have PBC.
//    // however, in no_domain_decomposition mode, ghost atoms are known to
//    // be shifted versions of local atoms.
//    n_nodes = atom->nlocal;
//  }
//  auto positions = torch::empty({n_nodes,3}, torch_float_dtype);
//  #pragma omp parallel for
//  for (int ii=0; ii<n_nodes; ++ii) {
//    int i = list->ilist[ii];
//    positions[i][0] = atom->x[i][0];
//    positions[i][1] = atom->x[i][1];
//    positions[i][2] = atom->x[i][2];
//  }
//
//  // ----- cell -----
//  auto cell = torch::zeros({3,3}, torch_float_dtype);
//  cell[0][0] = domain->h[0];
//  cell[0][1] = 0.0;
//  cell[0][2] = 0.0;
//  cell[1][0] = domain->h[5];
//  cell[1][1] = domain->h[1];
//  cell[1][2] = 0.0;
//  cell[2][0] = domain->h[4];
//  cell[2][1] = domain->h[3];
//  cell[2][2] = domain->h[2];
//
//  // ----- edge_index and unit_shifts -----
//  // count total number of edges
//  int n_edges = 0;
//  std::vector<int> n_edges_vec(n_nodes, 0);
//  #pragma omp parallel for reduction(+:n_edges)
//  for (int ii=0; ii<n_nodes; ++ii) {
//    int i = list->ilist[ii];
//    double xtmp = atom->x[i][0];
//    double ytmp = atom->x[i][1];
//    double ztmp = atom->x[i][2];
//    int *jlist = list->firstneigh[i];
//    int jnum = list->numneigh[i];
//    for (int jj=0; jj<jnum; ++jj) {
//      int j = jlist[jj];
//      j &= NEIGHMASK;
//      double delx = xtmp - atom->x[j][0];
//      double dely = ytmp - atom->x[j][1];
//      double delz = ztmp - atom->x[j][2];
//      double rsq = delx * delx + dely * dely + delz * delz;
//      if (rsq < r_max_squared) {
//        n_edges += 1;
//        n_edges_vec[ii] += 1;
//      }
//    }
//  }
//  // make first_edge vector to help with parallelizing following loop
//  std::vector<int> first_edge(n_nodes);
//  first_edge[0] = 0;
//  for (int ii=0; ii<n_nodes-1; ++ii) {
//    first_edge[ii+1] = first_edge[ii] + n_edges_vec[ii];
//  }
//  // fill edge_index and unit_shifts tensors
//  auto edge_index = torch::empty({2,n_edges}, torch::dtype(torch::kInt64));
//  auto unit_shifts = torch::zeros({n_edges,3}, torch_float_dtype);
//  auto shifts = torch::zeros({n_edges,3}, torch_float_dtype);
//  #pragma omp parallel for
//  for (int ii=0; ii<n_nodes; ++ii) {
//    int i = list->ilist[ii];
//    double xtmp = atom->x[i][0];
//    double ytmp = atom->x[i][1];
//    double ztmp = atom->x[i][2];
//    int *jlist = list->firstneigh[i];
//    int jnum = list->numneigh[i];
//    int k = first_edge[ii];
//    for (int jj=0; jj<jnum; ++jj) {
//      int j = jlist[jj];
//      j &= NEIGHMASK;
//      double delx = xtmp - atom->x[j][0];
//      double dely = ytmp - atom->x[j][1];
//      double delz = ztmp - atom->x[j][2];
//      double rsq = delx * delx + dely * dely + delz * delz;
//      if (rsq < r_max_squared) {
//        edge_index[0][k] = i;
//        if (domain_decomposition) {
//          edge_index[1][k] = j;
//        } else {
//          int j_local = atom->map(atom->tag[j]);
//          edge_index[1][k] = j_local;
//          double shiftx = atom->x[j][0] - atom->x[j_local][0];
//          double shifty = atom->x[j][1] - atom->x[j_local][1];
//          double shiftz = atom->x[j][2] - atom->x[j_local][2];
//          double shiftxs = std::round(domain->h_inv[0]*shiftx + domain->h_inv[5]*shifty + domain->h_inv[4]*shiftz);
//          double shiftys = std::round(domain->h_inv[1]*shifty + domain->h_inv[3]*shiftz);
//          double shiftzs = std::round(domain->h_inv[2]*shiftz);
//          unit_shifts[k][0] = shiftxs;
//          unit_shifts[k][1] = shiftys;
//          unit_shifts[k][2] = shiftzs;
//          shifts[k][0] = domain->h[0]*shiftxs + domain->h[5]*shiftys + domain->h[4]*shiftzs;
//          shifts[k][1] = domain->h[1]*shiftys + domain->h[3]*shiftzs;
//          shifts[k][2] = domain->h[2]*shiftzs;
//        }
//        k++;
//      }
//    }
//  }
//
//  // ----- node_attrs -----
//  // node_attrs is one-hot encoding for atomic numbers
//  auto mace_type = [this](int lammps_type) {
//    for (int i=0; i<mace_atomic_numbers.size(); ++i) {
//      if (mace_atomic_numbers[i]==lammps_atomic_numbers[lammps_type-1]) {
//        return i+1;
//      }
//    }
//    error->all(FLERR, "ERROR: problem converting lammps_type to mace_type.");
//    return -1;
//  };
//  int n_node_feats = mace_atomic_numbers.size();
//  auto node_attrs = torch::zeros({n_nodes,n_node_feats}, torch_float_dtype);
//  #pragma omp parallel for
//  for (int ii=0; ii<n_nodes; ++ii) {
//    int i = list->ilist[ii];
//    node_attrs[i][mace_type(atom->type[i])-1] = 1.0;
//  }
//
//  // ----- mask for ghost -----
//  auto mask = torch::zeros(n_nodes, torch::dtype(torch::kBool));
//  #pragma omp parallel for
//  for (int ii=0; ii<atom->nlocal; ++ii) {
//    int i = list->ilist[ii];
//    mask[i] = true;
//  }
//
//  auto batch = torch::zeros({n_nodes}, torch::dtype(torch::kInt64));
//  auto energy = torch::empty({1}, torch_float_dtype);
//  auto forces = torch::empty({n_nodes,3}, torch_float_dtype);
//  auto ptr = torch::empty({2}, torch::dtype(torch::kInt64));
//  auto weight = torch::empty({1}, torch_float_dtype);
//  ptr[0] = 0;
//  ptr[1] = n_nodes;
//  weight[0] = 1.0;
//
//  // pack the input, call the model, extract the output
//  c10::Dict<std::string, torch::Tensor> input;
//  input.insert("batch", batch);
//  input.insert("cell", cell);
//  input.insert("edge_index", edge_index);
//  input.insert("energy", energy);
//  input.insert("forces", forces);
//  input.insert("node_attrs", node_attrs);
//  input.insert("positions", positions);
//  input.insert("ptr", ptr);
//  input.insert("shifts", shifts);
//  input.insert("unit_shifts", unit_shifts);
//  input.insert("weight", weight);
//  auto output = model.forward({input, mask, true, true, false}).toGenericDict();
//
//  // mace energy
//  //   -> sum of site energies of local atoms
//  if (eflag_global) {
//    auto node_energy = output.at("node_energy").toTensor();
//    eng_vdwl = 0.0;
//    #pragma omp parallel for reduction(+:eng_vdwl)
//    for (int ii=0; ii<list->inum; ++ii) {
//      int i = list->ilist[ii];
//      eng_vdwl += node_energy[i].item<double>();
//    }
//  }
//
//  // mace forces
//  //   -> derivatives of total mace energy
//  forces = output.at("forces").toTensor();
//  #pragma omp parallel for
//  for (int ii=0; ii<list->inum; ++ii) {
//    int i = list->ilist[ii];
//    atom->f[i][0] = forces[i][0].item<double>();
//    atom->f[i][1] = forces[i][1].item<double>();
//    atom->f[i][2] = forces[i][2].item<double>();
//  }
//
//  // mace site energies
//  //   -> local atoms only
//  if (eflag_atom) {
//    auto node_energy = output.at("node_energy").toTensor();
//    #pragma omp parallel for
//    for (int ii=0; ii<list->inum; ++ii) {
//      int i = list->ilist[ii];
//      eatom[i] = node_energy[i].item<double>();
//    }
//  }
//
//  // mace virials (local atoms only)
//  //   -> derivatives of sum of site energies of local atoms
//  if (vflag_global) {
//    auto vir = output.at("virials").toTensor();
//    virial[0] = vir[0][0][0].item<double>();
//    virial[1] = vir[0][1][1].item<double>();
//    virial[2] = vir[0][2][2].item<double>();
//    virial[3] = 0.5*(vir[0][2][1].item<double>() + vir[0][1][2].item<double>());
//    virial[4] = 0.5*(vir[0][2][0].item<double>() + vir[0][0][2].item<double>());
//    virial[5] = 0.5*(vir[0][1][0].item<double>() + vir[0][0][1].item<double>());
//  }
//
//  // mace site virials
//  //   -> not available
//  if (vflag_atom) {
//    error->all(FLERR, "ERROR: pair_mace does not support vflag_atom.");
//  }

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMACEKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg == 1) {
    if (strcmp(arg[0], "no_domain_decomposition") == 0) {
      domain_decomposition = false;
    } else {
      error->all(FLERR, "Invalid option for pair_style mace.");
    }
  } else if (narg > 1) {
    error->all(FLERR, "Too many pair_style arguments for pair_style mace.");
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMACEKokkos<DeviceType>::coeff(int narg, char **arg)
{
  // TODO: remove print statements from this routine, or have a single proc print

  if (!allocated) allocate();

  std::cout << "Loading MACEKokkos model from \"" << arg[2] << "\" ...";
  model = torch::jit::load(arg[2]);
  std::cout << " finished." << std::endl;

  // extract default dtype from mace model
  for (auto p: model.named_attributes()) {
      // this is a somewhat random choice of variable to check. could it be improved?
      if (p.name == "model.node_embedding.linear.weight") {
          if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<float>()) {
            torch_float_dtype = torch::kFloat32;
          } else if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<double>()) {
            torch_float_dtype = torch::kFloat64;
          }
      }
  }
  std::cout << "  - The torch_float_dtype is: " << torch_float_dtype << std::endl;

  // extract r_max from mace model
  r_max = model.attr("r_max").toTensor().item<double>();
  r_max_squared = r_max*r_max;
  std::cout << "  - The r_max is: " << r_max << "." << std::endl;
  num_interactions = model.attr("num_interactions").toTensor().item<int64_t>();
  std::cout << "  - The model has: " << num_interactions << " layers." << std::endl;

  // extract atomic numbers from mace model
  auto a_n = model.attr("atomic_numbers").toTensor();
  for (int i=0; i<a_n.size(0); ++i) {
    mace_atomic_numbers.push_back(a_n[i].item<int64_t>());
  }
  std::cout << "  - The model atomic numbers are: " << mace_atomic_numbers << "." << std::endl;

  // extract atomic numbers from pair_coeff
  for (int i=3; i<narg; ++i) {
    auto iter = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    int index = std::distance(periodic_table.begin(), iter) + 1;
    lammps_atomic_numbers.push_back(index);
  }
  std::cout << "  - The pair_coeff atomic numbers are: " << lammps_atomic_numbers << "." << std::endl;

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

template<class DeviceType>
void PairMACEKokkos<DeviceType>::init_style()
{
//  if (force->newton_pair == 0) error->all(FLERR, "ERROR: Pair style mace requires newton pair on.");
//
//  /*
//    MACE requires the full neighbor list AND neighbors of ghost atoms
//    it appears that:
//      * without REQ_GHOST
//           list->gnum == 0
//           list->ilist does not include ghost atoms, but the jlists do
//      * with REQ_GHOST
//           list->gnum == atom->nghost
//           list->ilist includes ghost atoms
//  */
//  if (domain_decomposition) {
//    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
//  } else {
//    neighbor->add_request(this, NeighConst::REQ_FULL);
//  }

  PairMACE::init_style();

  // TODO: from lj_cut, possibly delete
  // adjust neighbor list request for KOKKOS

  //neighflag = lmp->kokkos->neighflag;
  //auto request = neighbor->find_request(this);
  //request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
  //                         !std::is_same<DeviceType,LMPDeviceType>::value);
  //request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  //if (neighflag == FULL) request->enable_full();
}

template<class DeviceType>
double PairMACEKokkos<DeviceType>::init_one(int i, int j)
{
//  // to account for message passing, require cutoff of n_layers * r_max
//  return num_interactions*model.attr("r_max").toTensor().item<double>();

  double cutone = PairMACE::init_one(i,j);

//  k_params.h_view(i,j).lj1 = lj1[i][j];
//  k_params.h_view(i,j).lj2 = lj2[i][j];
//  k_params.h_view(i,j).lj3 = lj3[i][j];
//  k_params.h_view(i,j).lj4 = lj4[i][j];
//  k_params.h_view(i,j).offset = offset[i][j];
//  k_params.h_view(i,j).cutsq = cutone*cutone;
//  k_params.h_view(j,i) = k_params.h_view(i,j);
//  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
//    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
//    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
//  }
//
//  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
//  k_cutsq.template modify<LMPHostType>();
//  k_params.template modify<LMPHostType>();

  return cutone;
}

template<class DeviceType>
void PairMACEKokkos<DeviceType>::allocate()
{
  PairMACE::allocate();

  // TODO: from lj_cut, possibly delete
  //int n = atom->ntypes;
  //memory->destroy(cutsq);
  //memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  //d_cutsq = k_cutsq.template view<DeviceType>();
  ////k_params = Kokkos::DualView<params_lj**,Kokkos::LayoutRight,DeviceType>("PairLJCut::params",n+1,n+1);
  //params = k_params.template view<DeviceType>();
}

namespace LAMMPS_NS {
template class PairMACEKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMACEKokkos<LMPHostType>;
#endif
}


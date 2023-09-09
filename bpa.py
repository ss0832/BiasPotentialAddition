import os
import sys
import glob
import copy
import time
import datetime
import psi4
import random
import math
import argparse
import itertools

from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import numpy as np

"""
    BiasPotentialAddition
    Copyright (C) 2023 ss0832

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
#please input psi4 inputfile.
(electronic charges) (spin multiply)
(element1) x y z
(element2) x y z
(element3) x y z
....
"""

"""
references(-opt):

RFO method
 The Journal of Physical Chemistry, Vol. 89, No. 1, 1985
FSB
 J. Chem. Phys. 1999, 111, 10806
Psi4
 D. G. A. Smith, L. A. Burns, A. C. Simmonett, R. M. Parrish, M. C. Schieber, R. Galvelis, P. Kraus, H. Kruse, R. Di Remigio, A. Alenaizan, A. M. James, S. Lehtola, J. P. Misiewicz, M. Scheurer, R. A. Shaw, J. B. Schriber, Y. Xie, Z. L. Glick, D. A. Sirianni, J. S. O'Brien, J. M. Waldrop, A. Kumar, E. G. Hohenstein, B. P. Pritchard, B. R. Brooks, H. F. Schaefer III, A. Yu. Sokolov, K. Patkowski, A. E. DePrince III, U. Bozkaya, R. A. King, F. A. Evangelista, J. M. Turney, T. D. Crawford, C. D. Sherrill, "Psi4 1.4: Open-Source Software for High-Throughput Quantum Chemistry", J. Chem. Phys. 152(18) 184108 (2020).
"""

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input psi4 files')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='300', help='iter. number')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument("-d", "--DELTA",  type=str, default='x', help='move step')

    parser.add_argument("-ma", "--manual_AFIR", nargs="*",  type=str, default=['0.0', '1', '2'], help='manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]')
    parser.add_argument("-rp", "--repulsive_potential", nargs="*",  type=str, default=['0.0','1.0', '1', '2'], help='Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]')
    parser.add_argument("-rpv2", "--repulsive_potential_v2", nargs="*",  type=str, default=['0.0','1.0','0.0','1','2','12','6', '1,2', '1-2'], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] ...]')
    parser.add_argument("-kp", "--keep_pot", nargs="*",  type=str, default=['0.0', '1.0', '1,2'], help='keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-akp", "--anharmonic_keep_pot", nargs="*",  type=str, default=['0.0', '1.0', '1.0', '1,2'], help='Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-ka", "--keep_angle", nargs="*",  type=str, default=['0.0', '90', '1,2,3'], help='keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] ')
    parser.add_argument("-kda", "--keep_dihedral_angle", nargs="*",  type=str, default=['0.0', '90', '1,2,3,4'], help='keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] ')
    parser.add_argument("-vpp", "--void_point_pot", nargs="*",  type=str, default=['0.0', '1.0', '0.0,0.0,0.0', '1',"2.0"], help='void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...] ')
    parser.add_argument("-gp", "--gaussian_pot", nargs="*",  type=str, default=['0.0'], help='Add Gaussian-type bias potential around the initial structure. (ex.) [energy (kJ/mol)]')
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-md", "--md_like_perturbation",  type=str, default="0.0", help='add perturbation like molecule dynamics (ex.) [[temperature (unit. K)]]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["AdaBelief"], help='optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE (quasi-Newton method) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    args = parser.parse_args()
    return args


def UFF_VDW_distance_lib(element):
    UFF_VDW_distance = {'H':2.886,'He':2.362 ,'Li' : 2.451 ,'Be': 2.745, 'B':4.083 ,'C': 3.851, 'N':3.660,'O':3.500 , 'F':3.364,'Ne': 3.243, 'Na':2.983,'Mg': 3.021 ,'Al':4.499 ,'Si': 4.295, 'P':4.147, 'S':4.035 ,'Cl':3.947,'Ar':3.868 ,'K':3.812 ,'Ca':3.399 ,'Sc':3.295 ,'Ti':3.175 ,'V': 3.144, 'Cr':3.023 ,'Mn': 2.961, 'Fe': 2.912,'Co':2.872 ,'Ni':2.834 ,'Cu':3.495 ,'Zn':2.763 ,'Ga': 4.383,'Ge':4.280,'As':4.230 ,'Se':4.205,'Br':4.189,'Kr':4.141 ,'Rb':4.114 ,'Sr': 3.641,'Y':3.345 ,'Zr':3.124 ,'Nb':3.165 ,'Mo':3.052 ,'Tc':2.998 ,'Ru':2.963 ,'Rh':2.929 ,'Pd':2.899 ,'Ag':3.148 ,'Cd':2.848 ,'In':4.463 ,'Sn':4.392 ,'Sb':4.420 ,'Te':4.470 , 'I':4.50, 'Xe':4.404 , 'Cs':4.517 ,'Ba':3.703 , 'La':3.522 , 'Ce':3.556 ,'Pr':3.606 ,'Nd':3.575 ,'Pm':3.547 ,'Sm':3.520 ,'Eu':3.493 ,'Gd':3.368 ,'Tb':3.451 ,'Dy':3.428 ,'Ho':3.409 ,'Er':3.391 ,'Tm':3.374 ,'Yb':3.355,'Lu':3.640 ,'Hf': 3.141,'Ta':3.170 ,'W':3.069 ,'Re':2.954 ,'Os':3.120 ,'Ir':2.840 ,'Pt':2.754 ,'Au':3.293 ,'Hg':2.705 ,'Tl':4.347 ,'Pb':4.297 ,'Bi':4.370 ,'Po':4.709 ,'At':4.750 ,'Rn': 4.765}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #ang.
                
    return UFF_VDW_distance[element] / UnitValueLib().bohr2angstroms#Bohr

            
            
def UFF_VDW_well_depth_lib(element):
                
    UFF_VDW_well_depth = {'H':0.044, 'He':0.056 ,'Li':0.025 ,'Be':0.085 ,'B':0.180,'C': 0.105, 'N':0.069, 'O':0.060,'F':0.050,'Ne':0.042 , 'Na':0.030, 'Mg':0.111 ,'Al':0.505 ,'Si': 0.402, 'P':0.305, 'S':0.274, 'Cl':0.227,  'Ar':0.185 ,'K':0.035 ,'Ca':0.238 ,'Sc':0.019 ,'Ti':0.017 ,'V':0.016 , 'Cr':0.015, 'Mn':0.013 ,'Fe': 0.013,'Co':0.014 ,'Ni':0.015 ,'Cu':0.005 ,'Zn':0.124 ,'Ga':0.415 ,'Ge':0.379, 'As':0.309 ,'Se':0.291,'Br':0.251,'Kr':0.220 ,'Rb':0.04 ,'Sr':0.235 ,'Y':0.072 ,'Zr':0.069 ,'Nb':0.059 ,'Mo':0.056 ,'Tc':0.048 ,'Ru':0.056 ,'Rh':0.053 ,'Pd':0.048 ,'Ag':0.036 ,'Cd':0.228 ,'In':0.599 ,'Sn':0.567 ,'Sb':0.449 ,'Te':0.398 , 'I':0.339,'Xe':0.332 , 'Cs':0.045 ,'Ba':0.364 , 'La':0.017 , 'Ce':0.013 ,'Pr':0.010 ,'Nd':0.010 ,'Pm':0.009 ,'Sm':0.008 ,'Eu':0.008 ,'Gd':0.009 ,'Tb':0.007 ,'Dy':0.007 ,'Ho':0.007 ,'Er':0.007 ,'Tm':0.006 ,'Yb':0.228 ,'Lu':0.041 ,'Hf':0.072 ,'Ta':0.081 ,'W':0.067 ,'Re':0.066 ,'Os':0.037 ,'Ir':0.073 ,'Pt':0.080 ,'Au':0.039 ,'Hg':0.385 ,'Tl':0.680 ,'Pb':0.663 ,'Bi':0.518 ,'Po':0.325 ,'At':0.284 ,'Rn':0.248}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 # kcal/mol
                
    return UFF_VDW_well_depth[element] / UnitValueLib().hartree2kcalmol
                
def covalent_radii_lib(element):
    CRL = {"H": 0.32, "He": 0.46, "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, "Na": 1.55, "Mg": 1.39, "Al":1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96, "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54,"Nb": 1.47,"Mo": 1.38,"Tc": 1.28,"Ru": 1.25,"Rh": 1.25,"Pd": 1.20,"Ag": 1.28,"Cd": 1.36,"In": 1.42,"Sn": 1.40,"Sb": 1.40,"Te": 1.36,"I": 1.33,"Xe": 1.31,"Cs": 2.32,"Ba": 1.96,"La":1.80,"Ce": 1.63,"Pr": 1.76,"Nd": 1.74,"Pm": 1.73,"Sm": 1.72,"Eu": 1.68,"Gd": 1.69 ,"Tb": 1.68,"Dy": 1.67,"Ho": 1.66,"Er": 1.65,"Tm": 1.64,"Yb": 1.70,"Lu": 1.62,"Hf": 1.52,"Ta": 1.46,"W": 1.37,"Re": 1.31,"Os": 1.29,"Ir": 1.22,"Pt": 1.23,"Au": 1.24,"Hg": 1.33,"Tl": 1.44,"Pb":1.44,"Bi":1.51,"Po":1.45,"At":1.47,"Rn":1.42}#ang.
    # ref. Pekka Pyykkö; Michiko Atsumi (2009). “Molecular single-bond covalent radii for elements 1 - 118”. Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. (H...Rn)
            
    return CRL[element] / UnitValueLib().bohr2angstroms#Bohr





class UnitValueLib: 
    def __init__(self):
        self.hartree2kcalmol = 627.509 #
        self.bohr2angstroms = 0.52917721067 #
        self.hartree2kjmol = 2625.500 #
        return


class CalculateMoveVector:
    def __init__(self, DELTA, Opt_params, Model_hess, FC_COUNT=-1, temperature=0.0):
        self.Opt_params = Opt_params 
        self.DELTA = DELTA
        self.Model_hess = Model_hess
        self.temperature = temperature
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
        self.FC_COUNT = FC_COUNT

    def calc_move_vector(self, iter, geom_num_list, new_g, opt_method_list, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector, e, pre_e, initial_geom_num_list):
        def update_trust_radii(trust_radii, dE, dE_predicted, displacement):
            if dE != 0:
                r =  dE_predicted / dE
            else:
                r = 1.0
            print("dE_predicted/dE : ",r)
            print("disp. - trust_radii :",abs(np.linalg.norm(displacement, ord=2) - trust_radii))
            if r < 0.25:
                return min(np.linalg.norm(displacement, ord=2) / 4, trust_radii / 4)
            elif r > 0.75 and abs(np.linalg.norm(displacement, ord=2) - trust_radii) < 1e-2:
                trust_radii *= 2.0
            else:
                pass
                    
            return np.clip(trust_radii, 0.001, 1.0)
            
            
        def TRM_FSB_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):#this function doesnt work well.

            #-----------------------
            if iter == 1:
                self.Model_hess = Model_hess_tmp(self.Model_hess.model_hess, momentum_disp=float(self.DELTA))#momentum_disp is trust_radii.
            
            trust_radii = self.Model_hess.momentum_disp
            
            aprrox_AFIR_e_shift = abs(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), pre_g.reshape(len(geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), self.Model_hess.model_hess),(geom_num_list - pre_geom).reshape( len(geom_num_list)*3,1)))
            
            
            
            #----------------------
            
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
    
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
            Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            new_hess = self.Model_hess.model_hess + delta_hess
            
            #---------------------- dogleg
            new_g_reshape = new_g.reshape(len(geom_num_list)*3, 1)
            ipsilon = 1e-8
            p_u = ((np.dot(new_g_reshape.T, new_g_reshape))/(np.dot(np.dot(new_g_reshape.T, new_hess), new_g_reshape)) + ipsilon)*new_g_reshape
            p_b = np.dot(np.linalg.inv(new_hess), new_g_reshape)
            if np.linalg.norm(p_u) >= trust_radii:
                move_vector = (trust_radii * (new_g_reshape/(np.linalg.norm(new_g_reshape) + ipsilon))).reshape(len(geom_num_list), 3)
            elif np.linalg.norm(p_b) <= trust_radii:
                move_vector = p_b.reshape(len(geom_num_list), 3)
            else:
                
                tau = np.sqrt((trust_radii ** 2 - np.linalg.norm(p_u) ** 2)/(np.linalg.norm(p_b - p_u) + ipsilon) ** 2)
                
                print(tau)
                
                if tau <= 1.0:
                    move_vector = (tau * p_u).reshape(len(geom_num_list), 3)
                else:
                    move_vector = ((2.0 - tau) * p_u + (tau - 1.0) * p_b).reshape(len(geom_num_list), 3)
                
            #---------------------
            
            move_vector = trust_radii * (move_vector/(np.linalg.norm(move_vector) + ipsilon))
            
            trust_radii = update_trust_radii(trust_radii, abs(AFIR_e - pre_AFIR_e), aprrox_AFIR_e_shift, geom_num_list - pre_geom)
            print("trust_radii: ",trust_radii)
            self.Model_hess = Model_hess_tmp(new_hess, momentum_disp=trust_radii)#valuable named 'momentum_disp' is trust_radii.
            return move_vector
    
        def TRM_BFGS_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):#this function doesnt work well.
 
            if iter == 1:
                self.Model_hess = Model_hess_tmp(self.Model_hess.model_hess, momentum_disp=float(self.DELTA))#momentum_disp is trust_radii.
            
            trust_radii = self.Model_hess.momentum_disp
           
           
            aprrox_AFIR_e_shift = abs(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), pre_g.reshape(len(geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), self.Model_hess.model_hess),(geom_num_list - pre_geom).reshape( len(geom_num_list)*3,1)))
            
           
            
            #----------------------
            
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            

            #print(Model_hess.model_hess)
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
            #print(A)

            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
     
            delta_hess = delta_hess_BFGS
            
                
            move_vector = (np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            aprrox_AFIR_e_shift = abs(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), pre_g.reshape(len(geom_num_list)*3, 1)) + 0.5 * np.dot(np.dot((geom_num_list - pre_geom).reshape(1, len(geom_num_list)*3), self.Model_hess.model_hess),(geom_num_list - pre_geom).reshape(len(geom_num_list)*3,1)))
            
            trust_radii = update_trust_radii(trust_radii, abs(AFIR_e - pre_AFIR_e), aprrox_AFIR_e_shift, geom_num_list - pre_geom)
            new_hess = self.Model_hess.model_hess + delta_hess
            
            #---------------------- dogleg
            new_g_reshape = new_g.reshape(len(geom_num_list)*3, 1)
            ipsilon = 1e-8
            p_u = ((np.dot(new_g_reshape.T, new_g_reshape))/(np.dot(np.dot(new_g_reshape.T, new_hess), new_g_reshape)) + ipsilon)*new_g_reshape
            p_b = np.dot(np.linalg.inv(new_hess), new_g_reshape)
            if np.linalg.norm(p_u) >= trust_radii:
                move_vector = (trust_radii*(new_g_reshape/(np.linalg.norm(new_g_reshape) + ipsilon))).reshape(len(geom_num_list), 3)
            elif np.linalg.norm(p_b) <= trust_radii:
                move_vector = p_b.reshape(len(geom_num_list), 3)
            else:
                tau = np.sqrt((trust_radii ** 2 - np.linalg.norm(p_u) ** 2)/(np.linalg.norm(p_b - p_u) + ipsilon) ** 2)
                if tau <= 1.0:
                    move_vector = (tau * p_u).reshape(len(geom_num_list), 3)
                else:
                    move_vector = ((2.0 - tau) * p_u + (tau - 1.0) * p_b).reshape(len(geom_num_list), 3)
                
            #---------------------
            trust_radii = update_trust_radii(trust_radii, abs(AFIR_e - pre_AFIR_e), aprrox_AFIR_e_shift, geom_num_list - pre_geom)
            print("trust_radii: ",trust_radii)
            self.Model_hess = Model_hess_tmp(new_hess, momentum_disp=trust_radii)#valuable named 'momentum_disp' is trust_radii.
            return move_vector
            
        def RFO_BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("RFO_BFGS_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            DELTA_for_QNM = self.DELTA
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
                
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector

            
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
            

            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))

            delta_hess = delta_hess_BFGS
            
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

                
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            DELTA_for_QNM = self.DELTA
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector
                
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM)
            self.Model_hess = Model_hess_tmp(new_hess)
            
            return move_vector
            
        def BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("BFGS_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            
          
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
           
            
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS
            else:
                new_hess = self.Model_hess.model_hess
                
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
              
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
        

        def RFO_FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("RFO_FSB_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            DELTA_for_QNM = self.DELTA
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
                
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector


            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
            Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

                
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            DELTA_for_QNM = self.DELTA
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
                
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM)
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
            
        def FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("FSB_quasi_newton_method")
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            
   
            A = delta_grad - np.dot(self.Model_hess.model_hess, displacement)
    
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
            delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, displacement) , displacement.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(displacement.T, self.Model_hess.model_hess), displacement))
            Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess)
            return move_vector
        
        # arXiv:2307.13744v1
        def momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("momentum_based_BFGS")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:

                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
                
            beta = 0.50
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
            
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
            
            
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS
            else:
                new_hess = self.Model_hess.model_hess
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            
            return move_vector 
        
        def momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("momentum_based_FSB")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.50
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
              
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
           
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
          
            
            
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, delta_momentum_disp) 
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))
            Bofill_const = np.dot(np.dot(np.dot(A.T, delta_momentum_disp), A.T), delta_momentum_disp) / np.dot(np.dot(np.dot(A.T, A), delta_momentum_disp.T), delta_momentum_disp)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess
            
            DELTA_for_QNM = self.DELTA
           
            
            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector 
            
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
                
          
        def RFO_momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("RFO_momentum_based_BFGS")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.5
            
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ", np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
          
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
           
            
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))

            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess_BFGS
            else:
                new_hess = self.Model_hess.model_hess
             
            DELTA_for_QNM = self.DELTA

            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
                
                return move_vector 
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
        
        def RFO_momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector):
            print("RFO_momentum_based_FSB")
            adam_count = self.Opt_params.adam_count
            if adam_count == 1:
                momentum_disp = pre_geom
                momentum_grad = pre_g
            else:
                momentum_disp = self.Model_hess.momentum_disp
                momentum_grad = self.Model_hess.momentum_grad
            beta = 0.5
            print("beta :", beta)
            delta_grad = (new_g - pre_g).reshape(len(geom_num_list)*3, 1)
            displacement = (geom_num_list - pre_geom).reshape(len(geom_num_list)*3, 1)
            
            if abs(displacement.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
              
                return move_vector
            if abs(np.nanmean(displacement)) < 1e-06:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too small!!")
                print("disp. avg.: ",np.nanmean(displacement))
                return move_vector
            new_momentum_disp = beta * momentum_disp + (1.0 - beta) * geom_num_list
            new_momentum_grad = beta * momentum_grad + (1.0 - beta) * new_g
            
            delta_momentum_disp = (new_momentum_disp - momentum_disp).reshape(len(geom_num_list)*3, 1)
            delta_momentum_grad = (new_momentum_grad - momentum_grad).reshape(len(geom_num_list)*3, 1)
            
            A = delta_momentum_grad - np.dot(self.Model_hess.model_hess, delta_momentum_disp)
            
            
            
            delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, delta_momentum_disp) 
            delta_hess_BFGS = (np.dot(delta_momentum_grad, delta_momentum_grad.T) / np.dot(delta_momentum_disp.T, delta_momentum_grad)) - (np.dot(np.dot(np.dot(self.Model_hess.model_hess, delta_momentum_disp) , delta_momentum_disp.T), self.Model_hess.model_hess.T)/ np.dot(np.dot(delta_momentum_disp.T, self.Model_hess.model_hess), delta_momentum_disp))
            Bofill_const = np.dot(np.dot(np.dot(A.T, delta_momentum_disp), A.T), delta_momentum_disp) / np.dot(np.dot(np.dot(A.T, A), delta_momentum_disp.T), delta_momentum_disp)
            delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS
            
            if iter % self.FC_COUNT != 0 or self.FC_COUNT == -1:
                new_hess = self.Model_hess.model_hess + delta_hess
            else:
                new_hess = self.Model_hess.model_hess

            DELTA_for_QNM = self.DELTA
            
            matrix_for_RFO = np.append(new_hess, new_g.reshape(len(geom_num_list)*3, 1), axis=1)
            tmp = np.array([np.append(new_g.reshape(1, len(geom_num_list)*3), 0.0)], dtype="float64")
            
            matrix_for_RFO = np.append(matrix_for_RFO, tmp, axis=0)
            eigenvalue, eigenvector = np.linalg.eig(matrix_for_RFO)
            lambda_for_calc = min(0.0, float(eigenvalue[np.argmin(eigenvalue)]))
            

            move_vector = (DELTA_for_QNM*np.dot(np.linalg.inv(new_hess - 0.1*lambda_for_calc*(np.eye(len(geom_num_list)*3))), new_g.reshape(len(geom_num_list)*3, 1))).reshape(len(geom_num_list), 3)
            
            
            if abs(move_vector.max()) > 2.0/self.bohr2angstroms:
                move_vector = 0.1*(new_g/np.linalg.norm(new_g))
                print("displacement is too large!!")
               
                return move_vector 
            print("lambda   : ",lambda_for_calc)
            print("step size: ",DELTA_for_QNM,"\n")
            self.Model_hess = Model_hess_tmp(new_hess, new_momentum_disp, new_momentum_grad)
            return move_vector 
                   
        #arXiv:1412.6980v9
        def AdaMax(geom_num_list, new_g):#not worked well
            print("AdaMax")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            if adam_count == 1:
                adamax_u = 1e-8
                self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, adamax_u)
                
            else:
                adamax_u = self.Opt_params.eve_d_tilde#eve_d_tilde = adamax_u 
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
            new_adamax_u = max(beta_v*adamax_u, np.linalg.norm(new_g))
               
            move_vector = []

            for i in range(len(geom_num_list)):
                move_vector.append((self.DELTA / (beta_m ** adam_count)) * (adam_m[i] / new_adamax_u))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, new_adamax_u)
            return move_vector
            
        #https://cs229.stanford.edu/proj2015/054_report.pdf
        def NAdam(geom_num_list, new_g):
            print("NAdam")
            mu = 0.975
            nu = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = []
            new_adam_v_hat = []
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(mu*adam_m[i] + (1.0 - mu)*(new_g[i]))
                new_adam_v[i] = copy.copy((nu*adam_v[i]) + (1.0 - nu)*(new_g[i]) ** 2)
                new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64") * ( mu / (1.0 - mu ** adam_count)) + np.array(new_g[i], dtype="float64") * ((1.0 - mu)/(1.0 - mu ** adam_count)))        
                new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64") * (nu / (1.0 - nu ** adam_count)))
            
            move_vector = []
            for i in range(len(geom_num_list)):
                move_vector.append( (self.DELTA*new_adam_m_hat[i]) / (np.sqrt(new_adam_v_hat[i] + Epsilon)))
                
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        
        
        #FIRE
        #Physical Review Letters, Vol. 97, 170201 (2006)
        def FIRE(geom_num_list, new_g):#MD-like optimization method. This method tends to converge local minima.
            print("FIRE")
            adam_count = self.Opt_params.adam_count
            N_acc = 5
            f_inc = 1.10
            f_acc = 0.99
            f_dec = 0.50
            dt_max = 0.8
            alpha_start = 0.1
            if adam_count == 1:
                self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, self.Opt_params.adam_v, 0, [0.1, alpha_start, 0])
                #valuable named 'eve_d_tilde' is parameters for FIRE.
                # [0]:dt [1]:alpha [2]:n_reset
            dt = self.Opt_params.eve_d_tilde[0]
            alpha = self.Opt_params.eve_d_tilde[1]
            n_reset = self.Opt_params.eve_d_tilde[2]
            
            pre_velocity = self.Opt_params.adam_v
            
            velocity = (1.0 - alpha) * pre_velocity + alpha * (np.linalg.norm(pre_velocity, ord=2)/np.linalg.norm(new_g, ord=2)) * new_g
            
            if adam_count > 1 and np.dot(pre_velocity.reshape(1, len(geom_num_list)*3), new_g.reshape(len(geom_num_list)*3, 1)) > 0:
                if n_reset > N_acc:
                    dt = min(dt * f_inc, dt_max)
                    alpha = alpha * f_acc
                n_reset += 1
            else:
                velocity *= 0.0
                alpha = alpha_start
                dt *= f_dec
                n_reset = 0
            
            velocity += dt*new_g
            
            move_vector = velocity * 0.0
            move_vector = copy.copy(dt * velocity)
            
            print("dt, alpha, n_reset :", dt, alpha, n_reset)
            self.Opt_params = Opt_calc_tmps(self.Opt_params.adam_m, velocity, adam_count, [dt, alpha, n_reset])
            return move_vector
        #RAdam
        #arXiv:1908.03265v4
        def RADAM(geom_num_list, new_g):
            print("RADAM")
            beta_m = 0.9
            beta_v = 0.99
            rho_inf = 2.0 / (1.0- beta_v) - 1.0 
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = []
            new_adam_v_hat = []
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy((beta_v*adam_v[i]) + (1.0-beta_v)*(new_g[i] - new_adam_m[i])**2) + Epsilon
                new_adam_m_hat.append(np.array(new_adam_m[i], dtype="float64")/(1.0-beta_m**adam_count))        
                new_adam_v_hat.append(np.array(new_adam_v[i], dtype="float64")/(1.0-beta_v**adam_count))
            rho = rho_inf - (2.0*adam_count*beta_v**adam_count)/(1.0 -beta_v**adam_count)
                        
            move_vector = []
            if rho > 4.0:
                l_alpha = []
                for j in range(len(new_adam_v)):
                    l_alpha.append(np.sqrt((abs(1.0 - beta_v**adam_count))/new_adam_v[j]))
                l_alpha = np.array(l_alpha, dtype="float64")
                r = np.sqrt(((rho-4.0)*(rho-2.0)*rho_inf)/((rho_inf-4.0)*(rho_inf-2.0)*rho))
                for i in range(len(geom_num_list)):
                    move_vector.append(self.DELTA*r*new_adam_m_hat[i]*l_alpha[i])
            else:
                for i in range(len(geom_num_list)):
                    move_vector.append(self.DELTA*new_adam_m_hat[i])
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #AdaBelief
        #ref. arXiv:2010.07468v5
        def AdaBelief(geom_num_list, new_g):
            print("AdaBelief")
            beta_m = 0.9
            beta_v = 0.99
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i]-new_adam_m[i])**2)
               
            move_vector = []

            for i in range(len(geom_num_list)):
                move_vector.append(self.DELTA*new_adam_m[i]/np.sqrt(new_adam_v[i]+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #AdaDiff
        #ref. https://iopscience.iop.org/article/10.1088/1742-6596/2010/1/012027/pdf  Dian Huang et al 2021 J. Phys.: Conf. Ser. 2010 012027
        def AdaDiff(geom_num_list, new_g, pre_g):
            print("AdaDiff")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2 + (1.0-beta_v) * (new_g[i] - pre_g[i]) ** 2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        
        #EVE
        #ref.arXiv:1611.01505v3
        def EVE(geom_num_list, new_g, AFIR_e, pre_AFIR_e, pre_g):
            print("EVE")
            beta_m = 0.9
            beta_v = 0.999
            beta_d = 0.999
            c = 10
            Epsilon = 1e-08 
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            if adam_count > 1:
                eve_d_tilde = self.Opt_params.eve_d_tilde
            else:
                eve_d_tilde = 1.0
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i])/(1 - beta_v**adam_count))
                
            if adam_count > 1:
                eve_d = abs(AFIR_e - pre_AFIR_e)/ min(AFIR_e, pre_AFIR_e)
                eve_d_hat = np.clip(eve_d, 1/c , c)
                eve_d_tilde = beta_d*eve_d_tilde + (1.0 - beta_d)*eve_d_hat
                
            else:
                pass
            
            for i in range(len(geom_num_list)):
                 move_vector.append((self.DELTA/eve_d_tilde)*new_adam_m_hat[i]/(np.sqrt(new_adam_v_hat[i])+Epsilon))
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, eve_d_tilde)
            return move_vector
        #AdamW
        #arXiv:2302.06675v4
        def AdamW(geom_num_list, new_g):
            print("AdamW")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            AdamW_lambda = 0.001
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon) + AdamW_lambda * geom_num_list[i])
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #Adam
        #arXiv:1412.6980
        def Adam(geom_num_list, new_g):
            print("Adam")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            
            new_adam_m_hat = adam_m*0.0
            new_adam_v_hat = adam_v*0.0
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i])**2)
     
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_m_hat[i] = copy.copy(new_adam_m[i]/(1 - beta_m**adam_count))
                new_adam_v_hat[i] = copy.copy((new_adam_v[i] + Epsilon)/(1 - beta_v**adam_count))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_adam_m_hat[i]/np.sqrt(new_adam_v_hat[i]+Epsilon))
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        #Adafactor
        #arXiv:1804.04235v1
        def Adafactor(geom_num_list, new_g):
            print("Adafactor")
            Epsilon_1 = 1e-08
            Epsilon_2 = self.DELTA

            adam_count = self.Opt_params.adam_count
            beta = 1 - adam_count ** (-0.8)
            rho = min(0.01, 1/np.sqrt(adam_count))
            alpha = max(np.sqrt(np.square(geom_num_list).mean()),  Epsilon_2) * rho
            adam_v = self.Opt_params.adam_m
            adam_u = self.Opt_params.adam_v
            new_adam_v = adam_v*0.0
            new_adam_u = adam_u*0.0
            new_adam_v = adam_v*0.0
            new_adam_u = adam_u*0.0
            new_adam_u_hat = adam_u*0.0
            for i in range(len(geom_num_list)):
                new_adam_v[i] = copy.copy(beta*adam_v[i] + (1.0-beta)*((new_g[i])**2 + np.array([1,1,1]) * Epsilon_1))
                new_adam_u[i] = copy.copy(new_g[i]/np.sqrt(new_adam_v[i]))
                
                        
            move_vector = []
            for i in range(len(geom_num_list)):
                new_adam_u_hat[i] = copy.copy(new_adam_u[i] / max(1, np.sqrt(np.square(new_adam_u).mean())))
            
            
            for i in range(len(geom_num_list)):
                 move_vector.append(alpha*new_adam_u_hat[i])
                 
            self.Opt_params = Opt_calc_tmps(new_adam_v, new_adam_u, adam_count)
        
            return move_vector
        #Prodigy
        #arXiv:2306.06101v1
        def Prodigy(geom_num_list, new_g, initial_geom_num_list):
            print("Prodigy")
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            adam_count = self.Opt_params.adam_count

            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            if adam_count == 1:
                adam_r = 0.0
                adam_s = adam_m*0.0
                d = 1e-1
                new_d = d
                self.Opt_params = Opt_calc_tmps(adam_m, adam_v, adam_count - 1, [d, adam_r, adam_s])
            else:
                d = self.Opt_params.eve_d_tilde[0]
                adam_r = self.Opt_params.eve_d_tilde[1]
                adam_s = self.Opt_params.eve_d_tilde[2]
                
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            new_adam_s = adam_s*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]*d))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(new_g[i]*d)**2)
                
                new_adam_s[i] = np.sqrt(beta_v)*adam_s[i] + (1.0 - np.sqrt(beta_v))*self.DELTA*new_g[i]*d**2  
            new_adam_r = np.sqrt(beta_v)*adam_r + (1.0 - np.sqrt(beta_v))*(np.dot(new_g.reshape(1,len(new_g)*3), (initial_geom_num_list - geom_num_list).reshape(len(new_g)*3,1)))*self.DELTA*d**2
            
            new_d = float(max((new_adam_r / np.linalg.norm(new_adam_s ,ord=1)), d))
            move_vector = []

            for i in range(len(geom_num_list)):
                 move_vector.append(self.DELTA*new_d*new_adam_m[i]/(np.sqrt(new_adam_v[i])+Epsilon*d))
            
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count, [new_d, new_adam_r, new_adam_s])
            return move_vector
        
        #AdaBound
        #arXiv:1902.09843v1
        def Adabound(geom_num_list, new_g):
            print("AdaBound")
            adam_count = self.Opt_params.adam_count
            move_vector = []
            beta_m = 0.9
            beta_v = 0.999
            Epsilon = 1e-08
            if adam_count == 1:
                adam_m = self.Opt_params.adam_m
                adam_v = np.zeros((len(geom_num_list),3,3))
            else:
                adam_m = self.Opt_params.adam_m
                adam_v = self.Opt_params.adam_v
                
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            V = adam_m*0.0
            Eta = adam_m*0.0
            Eta_hat = adam_m*0.0
            
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(beta_m*adam_m[i] + (1.0-beta_m)*(new_g[i]))
                new_adam_v[i] = copy.copy(beta_v*adam_v[i] + (1.0-beta_v)*(np.dot(np.array([new_g[i]]).T, np.array([new_g[i]]))))
                V[i] = copy.copy(np.diag(new_adam_v[i]))
                
                Eta_hat[i] = copy.copy(np.clip(self.DELTA/np.sqrt(V[i]), 0.1 - (0.1/(1.0 - beta_v) ** (adam_count + 1)) ,0.1 + (0.1/(1.0 - beta_v) ** adam_count) ))
                Eta[i] = copy.copy(Eta_hat[i]/np.sqrt(adam_count))
                    
            for i in range(len(geom_num_list)):
                move_vector.append(Eta[i] * new_adam_m[i])
            
            return move_vector    
        
        #Adadelta
        #arXiv:1212.5701v1
        def Adadelta(geom_num_list, new_g):#delta is not required. This method tends to converge local minima.
            print("Adadelta")
            rho = 0.9
            adam_count = self.Opt_params.adam_count
            adam_m = self.Opt_params.adam_m
            adam_v = self.Opt_params.adam_v
            new_adam_m = adam_m*0.0
            new_adam_v = adam_v*0.0
            Epsilon = 1e-06
            for i in range(len(geom_num_list)):
                new_adam_m[i] = copy.copy(rho * adam_m[i] + (1.0 - rho)*(new_g[i]) ** 2)
            move_vector = []
            
            for i in range(len(geom_num_list)):
                if adam_count > 1:
                    move_vector.append(new_g[i] * (np.sqrt(np.square(adam_v).mean()) + Epsilon)/(np.sqrt(np.square(new_adam_m).mean()) + Epsilon))
                else:
                    move_vector.append(new_g[i])
            if abs(np.sqrt(np.square(move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(new_g).mean())) > self.RMS_FORCE_THRESHOLD:
                move_vector = new_g

            for i in range(len(geom_num_list)):
                new_adam_v[i] = copy.copy(rho * adam_v[i] + (1.0 - rho) * (move_vector[i]) ** 2)
                 
            self.Opt_params = Opt_calc_tmps(new_adam_m, new_adam_v, adam_count)
            return move_vector
        

        def MD_like_Perturbation(move_vector):#This function is just for fun. Thus, it is no scientific basis.
            """Langevin equation"""
            Boltzmann_constant = 3.16681*10**(-6) # hartree/K
            damping_coefficient = 10.0
	
            temperature = self.temperature
            perturbation = self.DELTA * np.sqrt(2.0 * damping_coefficient * Boltzmann_constant * temperature) * np.random.normal(loc=0.0, scale=1.0, size=3*len(move_vector)).reshape(len(move_vector), 3)

            return perturbation

        move_vector_list = []
     
        #---------------------------------
        
        for opt_method in opt_method_list:
            # group of steepest descent
            if opt_method == "RADAM":
                tmp_move_vector = RADAM(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adam":
                tmp_move_vector = Adam(geom_num_list, new_g) 
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adadelta":
                tmp_move_vector = Adadelta(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdamW":
                tmp_move_vector = AdamW(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaDiff":
                tmp_move_vector = AdaDiff(geom_num_list, new_g, pre_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adafactor":
                tmp_move_vector = Adafactor(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaBelief":
                tmp_move_vector = AdaBelief(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Adabound":
                tmp_move_vector = Adabound(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "EVE":
                tmp_move_vector = EVE(geom_num_list, new_g, AFIR_e, pre_AFIR_e, pre_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "Prodigy":
                tmp_move_vector = Prodigy(geom_num_list, new_g, pre_geom)#initial_geom_num_list is not assigned.
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "AdaMax":
                tmp_move_vector = AdaMax(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "NAdam":    
                tmp_move_vector = NAdam(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            elif opt_method == "FIRE":
                tmp_move_vector = FIRE(geom_num_list, new_g)
                move_vector_list.append(tmp_move_vector)
                
            # group of quasi-Newton method
            
            elif opt_method == "TRM_FSB":
                if iter != 0:
                    tmp_move_vector = TRM_FSB_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "TRM_BFGS":
                if iter != 0:
                    tmp_move_vector = TRM_BFGS_dogleg_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
            
            elif opt_method == "BFGS":
                if iter != 0:
                    tmp_move_vector = BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_BFGS":
                if iter != 0:
                    tmp_move_vector = RFO_BFGS_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "FSB":
                if iter != 0:
                    tmp_move_vector = FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                    
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
                    
            elif opt_method == "RFO_FSB":
                if iter != 0:
                    tmp_move_vector = RFO_FSB_quasi_newton_method(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "mBFGS":
                if iter != 0:
                    tmp_move_vector = momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "mFSB":
                if iter != 0:
                    tmp_move_vector = momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mBFGS":
                if iter != 0:
                    tmp_move_vector = RFO_momentum_based_BFGS(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            elif opt_method == "RFO_mFSB":
                if iter != 0:
                    tmp_move_vector = RFO_momentum_based_FSB(geom_num_list, new_g, pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector)
                    move_vector_list.append(tmp_move_vector)
                else:
                    tmp_move_vector = 0.01*new_g 
                    move_vector_list.append(tmp_move_vector)
            else:
                print("optimization method that this program is not sppourted is selected... thus, default method is selected.")
                tmp_move_vector = AdaBelief(geom_num_list, new_g)
        #---------------------------------
        
        if len(move_vector_list) > 1:
            if abs(new_g.max()) < self.MAX_FORCE_SWITCHING_THRESHOLD and abs(np.sqrt(np.square(new_g).mean())) < self.RMS_FORCE_SWITCHING_THRESHOLD:
                move_vector = copy.copy(move_vector_list[1])
                print("Chosen method: ", opt_method_list[1])
            else:
                move_vector = copy.copy(move_vector_list[0])
                print("Chosen method: ", opt_method_list[0])
        else:
            move_vector = copy.copy(move_vector_list[0])
        
        perturbation = MD_like_Perturbation(move_vector)
        
        move_vector += perturbation
        print("perturbation: ", np.linalg.norm(perturbation))
        
        print("step radii: ", np.linalg.norm(move_vector))
        
        hess_eigenvalue, _ = np.linalg.eig(self.Model_hess.model_hess)
        
        print("NORMAL MODE EIGENVALUE:\n",np.sort(hess_eigenvalue),"\n")
        
        #---------------------------------
        new_geometry = (geom_num_list - move_vector) * self.bohr2angstroms
        
        return new_geometry, np.array(move_vector, dtype="float64"), iter, self.Opt_params, self.Model_hess
        


class BiasPotentialAddtion:#this class is GOD class, so this class isn't good.
    def __init__(self, args):
    
        UVL = UnitValueLib()
        np.set_printoptions(precision=12, floatmode="fixed", suppress=True)
        self.hartree2kcalmol = UVL.hartree2kcalmol #
        self.bohr2angstroms = UVL.bohr2angstroms #
        self.hartree2kjmol = UVL.hartree2kjmol #
 
        self.ENERGY_LIST_FOR_PLOTTING = [] #
        self.AFIR_ENERGY_LIST_FOR_PLOTTING = [] #
        self.NUM_LIST = [] #

        self.MAX_FORCE_THRESHOLD = 0.0003 #
        self.RMS_FORCE_THRESHOLD = 0.0002 #
        self.MAX_DISPLACEMENT_THRESHOLD = 0.0015 # 
        self.RMS_DISPLACEMENT_THRESHOLD = 0.0010 #
        self.MAX_FORCE_SWITCHING_THRESHOLD = 0.0010
        self.RMS_FORCE_SWITCHING_THRESHOLD = 0.0008
        
        self.args = args #
        self.FC_COUNT = args.calc_exact_hess # 
        #---------------------------
        self.temperature = float(args.md_like_perturbation)
        
        #---------------------------
        if len(args.opt_method) > 2:
            print("invaild input (-opt)")
            sys.exit(0)
        
        if args.DELTA == "x":
            if args.opt_method[0] == "FSB":
                args.DELTA = 0.15
            elif args.opt_method[0] == "RFO_FSB":
                args.DELTA = 0.15
            elif args.opt_method[0] == "BFGS":
                args.DELTA = 0.15
            elif args.opt_method[0] == "RFO_BFGS":
                args.DELTA = 0.15
                
            elif args.opt_method[0] == "mBFGS":
                args.DELTA = 0.50
            elif args.opt_method[0] == "mFSB":
                args.DELTA = 0.50
            elif args.opt_method[0] == "RFO_mBFGS":
                args.DELTA = 0.30
            elif args.opt_method[0] == "RFO_mFSB":
                args.DELTA = 0.30

            elif args.opt_method[0] == "Adabound":
                args.DELTA = 0.01
            elif args.opt_method[0] == "AdaMax":
                args.DELTA = 0.01
                
            elif args.opt_method[0] == "TRM_FSB":
                args.DELTA = 0.60
            elif args.opt_method[0] == "TRM_BFGS":
                args.DELTA = 0.60
            else:
                args.DELTA = 0.06
        else:
            pass 
        self.DELTA = float(args.DELTA) # 

        self.N_THREAD = args.N_THREAD #
        self.SET_MEMORY = args.SET_MEMORY #
        self.START_FILE = args.INPUT #
        self.NSTEP = args.NSTEP #
        #-----------------------------
        self.BASIS_SET = args.basisset # 
        self.FUNCTIONAL = args.functional # 
        
        if len(args.sub_basisset) % 2 != 0:
            print("invaild input (-sub_bs)")
            sys.exit(0)
        
        self.SUB_BASIS_SET = "" # 
        if len(args.sub_basisset) > 0:
            self.SUB_BASIS_SET +="\nassign "+str(self.BASIS_SET)+"\n" # 
            for j in range(int(len(args.sub_basisset)/2)):
                self.SUB_BASIS_SET += "assign "+args.sub_basisset[2*j]+" "+args.sub_basisset[2*j+1]+"\n"
            print("Basis Sets defined by User are detected.")
            print(self.SUB_BASIS_SET) #
        #-----------------------------
        
        self.BPA_FOLDER_DIRECTORY = str(datetime.datetime.now().date())+"/"+self.START_FILE[:-4]+"_BPA_"+self.FUNCTIONAL+"_"+self.BASIS_SET+"_"+str(time.time())+"/"
        os.makedirs(self.BPA_FOLDER_DIRECTORY, exist_ok=True) #
        
        self.Model_hess = None #
        self.Opt_params = None #
        
        return
        
    def make_geometry_list(self):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """Load initial structure"""
        geometry_list = []
 
        with open(self.START_FILE,"r") as f:
            words = f.readlines()
            
        start_data = []
        for word in words:
            start_data.append(word.split())
            
        electric_charge_and_multiplicity = start_data[0]
        element_list = []
            


        for i in range(1, len(start_data)):
            element_list.append(start_data[i][0])
                
        geometry_list.append(start_data)


        return geometry_list, element_list, electric_charge_and_multiplicity

    def make_geometry_list_2(self, new_geometry, element_list, electric_charge_and_multiplicity):#numbering name of function is not good. (ex. function_1, function_2, ...) 
        """load structure updated geometry for next QM calculation"""
        new_geometry = new_geometry.tolist()
        
        geometry_list = []

        new_data = [electric_charge_and_multiplicity]
        for num, geometry in enumerate(new_geometry):
           
            geometry = list(map(str, geometry))
            geometry = [element_list[num]] + geometry
            new_data.append(geometry)
            print(" ".join(geometry))
            
        geometry_list.append(new_data)
        return geometry_list

    def make_psi4_input_file(self, geometry_list, iter):
        """structure updated geometry is saved."""
        file_directory = self.BPA_FOLDER_DIRECTORY+"samples_"+str(self.START_FILE[:-4])+"_"+str(iter)
        try:
            os.mkdir(file_directory)
        except:
            pass
        for y, geometry in enumerate(geometry_list):
            with open(file_directory+"/"+self.START_FILE[:-4]+"_"+str(y)+".xyz","w") as w:
                for rows in geometry:
                    for row in rows:
                        w.write(str(row))
                        w.write(" ")
                    w.write("\n")
        return file_directory

    def sinple_plot(self, num_list, energy_list, energy_list_2, file_directory):
        
        fig = plt.figure()

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.plot(num_list, energy_list, "g--.")
        ax2.plot(num_list, energy_list_2, "b--.")

        ax1.set_xlabel('ITR.')
        ax2.set_xlabel('ITR.')

        ax1.set_ylabel('Gibbs Energy [kcal/mol]')
        ax2.set_ylabel('Gibbs Energy [kcal/mol]')
        plt.title('normal_above AFIR_below')
        plt.tight_layout()
        plt.savefig(self.BPA_FOLDER_DIRECTORY+"Energy_plot_sinple_"+str(time.time())+".png", format="png", dpi=300)
        plt.close()
        return

    def xyz_file_make(self):
        """optimized path is saved."""
        print("\ngeometry collection processing...\n")
        file_list = glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9]/*.xyz") + glob.glob(self.BPA_FOLDER_DIRECTORY+"samples_*_[0-9][0-9][0-9][0-9][0-9][0-9]/*.xyz")  
        #print(file_list,"\n")
        for m, file in enumerate(file_list):
            #print(file,m)
            with open(file,"r") as f:
                sample = f.readlines()
                with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w:
                    atom_num = len(sample)-1
                    w.write(str(atom_num)+"\n")
                    w.write("Frame "+str(m)+"\n")
                del sample[0]
                for i in sample:
                    with open(self.BPA_FOLDER_DIRECTORY+self.START_FILE[:-4]+"_collection.xyz","a") as w2:
                        w2.write(i)
        print("\ngeometry collection is completed...\n")
        return

    def psi4_calculation(self, file_directory, element_list, electric_charge_and_multiplicity, iter):
        """execute QM calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                print("\n",input_file,"\n")
                if int(electric_charge_and_multiplicity[1]) > 1:
                    psi4.set_options({'reference': 'uks'})
                logfile = file_directory+"/"+self.START_FILE[:-4]+'_'+str(num)+'.log'
                psi4.set_options({"MAXITER": 700})
                if len(self.SUB_BASIS_SET) > 0:
                    psi4.basis_helper(self.SUB_BASIS_SET, name='User_Basis_Set', set_option=False)
                    psi4.set_options({"basis":'User_Basis_Set'})
                else:
                    psi4.set_options({"basis":self.BASIS_SET})
                
                psi4.set_output_file(logfile)
                psi4.set_num_threads(nthread=self.N_THREAD)
                psi4.set_memory(self.SET_MEMORY)
                
                psi4.set_options({"cubeprop_tasks": ["esp"],'cubeprop_filepath': file_directory})
                with open(input_file,"r") as f:
                    input_data = f.read()
                    input_data = psi4.geometry(input_data)
                    input_data_for_display = np.array(input_data.geometry(), dtype = "float64")
                            
                g, wfn = psi4.gradient(self.FUNCTIONAL, molecule=input_data, return_wfn=True)

                g = np.array(g, dtype = "float64")
                psi4.oeprop(wfn, 'DIPOLE')
                psi4.oeprop(wfn, 'MULLIKEN_CHARGES')
                psi4.oeprop(wfn, 'LOWDIN_CHARGES')
                #psi4.oeprop(wfn, 'WIBERG_LOWDIN_INDICES')
                lumo_alpha = wfn.nalpha()
                lumo_beta = wfn.nbeta()

                MO_levels =np.array(wfn.epsilon_a_subset("AO","ALL")).tolist()#MO energy levels
                with open(self.BPA_FOLDER_DIRECTORY+"MO_levels.csv" ,"a") as f:
                    f.write(",".join(list(map(str,MO_levels))+[str(lumo_alpha),str(lumo_beta)])+"\n")
                with open(self.BPA_FOLDER_DIRECTORY+"dipole.csv" ,"a") as f:
                    f.write(",".join(list(map(str,(psi4.constants.dipmom_au2debye*wfn.variable('DIPOLE')).tolist()))+[str(np.linalg.norm(psi4.constants.dipmom_au2debye*wfn.variable('DIPOLE'),ord=2))])+"\n")
                with open(self.BPA_FOLDER_DIRECTORY+"MULLIKEN_CHARGES.csv" ,"a") as f:
                    f.write(",".join(list(map(str,wfn.variable('MULLIKEN CHARGES').tolist())))+"\n")           
                #with open(input_file[:-4]+"_WIBERG_LOWDIN_INDICES.csv" ,"a") as f:
                #    for i in range(len(np.array(wfn.variable('WIBERG LOWDIN INDICES')).tolist())):
                #        f.write(",".join(list(map(str,np.array(wfn.variable('WIBERG LOWDIN INDICES')).tolist()[i])))+"\n")           
                        
                with open(input_file[:-4]+".log","r") as f:
                    word_list = f.readlines()
                    for word in word_list:
                        if "    Total Energy =             " in word:
                            word = word.replace("    Total Energy =             ","")
                            e = (float(word))
                print("\n")

                
                if self.FC_COUNT == -1:
                    pass
                
                elif iter % self.FC_COUNT == 0:
                    
                    """exact hessian"""
                    _, wfn = psi4.frequencies(self.FUNCTIONAL, return_wfn=True, ref_gradient=wfn.gradient())
                    exact_hess = np.array(wfn.hessian())
                    
                    freqs = np.array(wfn.frequencies())
                    
                    print("frequencies: \n",freqs)
                    self.Model_hess = Model_hess_tmp(exact_hess, momentum_disp=self.Model_hess.momentum_disp, momentum_grad=self.Model_hess.momentum_grad)
                
                


            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return 0, 0, 0, finish_frag 
                
            psi4.core.clean() 
        return e, g, input_data_for_display, finish_frag


    def calc_biaspot(self, e, g, geom_num_list, element_list,  force_data, pre_g, iter, initial_geom_num_list):
        numerical_derivative_delta = 0.0001 #unit:Bohr
        #g:hartree/Bohr
        #e:hartree
        #geom_num_list:Bohr
        
     
        
        def calc_LJ_Repulsive_pot(geom_num_list, well_scale , dist_scale, fragm_1, fragm_2, element_list):  
            energy = 0.0
            for i, j in itertools.product(fragm_1, fragm_2):
                UFF_VDW_well_depth = np.sqrt(well_scale*UFF_VDW_well_depth_lib(element_list[i-1]) + well_scale*UFF_VDW_well_depth_lib(element_list[j-1]))
                UFF_VDW_distance = np.sqrt(UFF_VDW_distance_lib(element_list[i-1])*dist_scale + UFF_VDW_distance_lib(element_list[j-1])*dist_scale)
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                energy += UFF_VDW_well_depth * ( -2 * ( UFF_VDW_distance / vector ) ** 6 + ( UFF_VDW_distance / vector ) ** 12)
                
            return energy
        
            
        def calc_LJ_Repulsive_pot_grad(geom_num_list, well_scale , dist_scale, fragm_1, fragm_2, element_list):
            grad = geom_num_list*0.0
            for i, j in itertools.product(fragm_1, fragm_2):
                UFF_VDW_well_depth = np.sqrt(well_scale*UFF_VDW_well_depth_lib(element_list[i-1]) + well_scale*UFF_VDW_well_depth_lib(element_list[j-1]))
                UFF_VDW_distance = np.sqrt(UFF_VDW_distance_lib(element_list[i-1])*dist_scale + UFF_VDW_distance_lib(element_list[j-1])*dist_scale)
                
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                
                grad_x_1 = 12 * UFF_VDW_well_depth * (( ( UFF_VDW_distance ** 6 / vector ** 8 ) - ( UFF_VDW_distance ** 12 / vector ** 14 ) ) * (geom_num_list[i-1][0] - geom_num_list[j-1][0])) 
                grad_y_1 = 12 * UFF_VDW_well_depth * (( ( UFF_VDW_distance ** 6 / vector ** 8 ) - ( UFF_VDW_distance ** 12 / vector ** 14 ) ) * (geom_num_list[i-1][1] - geom_num_list[j-1][1])) 
                grad_z_1 = 12 * UFF_VDW_well_depth * (( ( UFF_VDW_distance ** 6 / vector ** 8 ) - ( UFF_VDW_distance ** 12 / vector ** 14 ) ) * (geom_num_list[i-1][2] - geom_num_list[j-1][2])) 

                grad[i-1] += np.array([grad_x_1,grad_y_1,grad_z_1], dtype="float64") #hartree/Bohr
                grad[j-1] += np.array([-1.0 * grad_x_1, -1.0 * grad_y_1, -1.0 * grad_z_1], dtype="float64") #hartree/Bohr
                
            return grad
   
   
        def calc_LJ_Repulsive_pot_hess(geom_num_list, well_scale , dist_scale, fragm_1, fragm_2, element_list, hessian):
            
            for i, j in itertools.product(fragm_1, fragm_2):
                tmp_hess = hessian*0.0
                UFF_VDW_well_depth = np.sqrt(well_scale*UFF_VDW_well_depth_lib(element_list[i-1]) + well_scale*UFF_VDW_well_depth_lib(element_list[j-1]))
                UFF_VDW_distance = np.sqrt(UFF_VDW_distance_lib(element_list[i-1])*dist_scale + UFF_VDW_distance_lib(element_list[j-1])*dist_scale)
                
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                
                hessian_x1x1 = 12 * UFF_VDW_well_depth * ((geom_num_list[i-1][0] - geom_num_list[j-1][0]) ** 2 * (-8 * (UFF_VDW_distance ** 6 / vector ** 10) + 14 * (UFF_VDW_distance ** 12 / vector ** 16)) + ((UFF_VDW_distance ** 6 / vector ** 8) - (UFF_VDW_distance ** 12 / vector ** 14)))
                hessian_x1y1 = 12 * UFF_VDW_well_depth * ((geom_num_list[i-1][0] - geom_num_list[j-1][0]) * (geom_num_list[i-1][1] - geom_num_list[j-1][1]) * (-8 * (UFF_VDW_distance ** 6 / vector ** 10) + 14 * (UFF_VDW_distance ** 12 / vector ** 16)))
                hessian_x1z1 = 12 * UFF_VDW_well_depth * ((geom_num_list[i-1][0] - geom_num_list[j-1][0]) * (geom_num_list[i-1][2] - geom_num_list[j-1][2]) * (-8 * (UFF_VDW_distance ** 6 / vector ** 10) + 14 * (UFF_VDW_distance ** 12 / vector ** 16)))
                hessian_x1x2 = -1 * hessian_x1x1
                hessian_x1y2 = -1 * hessian_x1y1
                hessian_x1z2 = -1 * hessian_x1z1
                
                hessian_y1x1 = hessian_x1y1
                hessian_y1y1 = 12 * UFF_VDW_well_depth * ((geom_num_list[i-1][1] - geom_num_list[j-1][1]) ** 2 * (-8 * (UFF_VDW_distance ** 6 / vector ** 10) + 14 * (UFF_VDW_distance ** 12 / vector ** 16)) + ((UFF_VDW_distance ** 6 / vector ** 8) - (UFF_VDW_distance ** 12 / vector ** 14)))
                hessian_y1z1 = 12 * UFF_VDW_well_depth * ((geom_num_list[i-1][1] - geom_num_list[j-1][1]) * (geom_num_list[i-1][2] - geom_num_list[j-1][2]) * (-8 * (UFF_VDW_distance ** 6 / vector ** 10) + 14 * (UFF_VDW_distance ** 12 / vector ** 16)))
                hessian_y1x2 = -1 * hessian_y1x1
                hessian_y1y2 = -1 * hessian_y1y1
                hessian_y1z2 = -1 * hessian_y1z1
                
                hessian_z1x1 = hessian_x1z1
                hessian_z1y1 = hessian_y1z1
                hessian_z1z1 = 12 * UFF_VDW_well_depth * ((geom_num_list[i-1][2] - geom_num_list[j-1][2]) ** 2 * (-8 * (UFF_VDW_distance ** 6 / vector ** 10) + 14 * (UFF_VDW_distance ** 12 / vector ** 16)) + ((UFF_VDW_distance ** 6 / vector ** 8) - (UFF_VDW_distance ** 12 / vector ** 14)))
                hessian_z1x2 = -1 * hessian_z1x1
                hessian_z1y2 = -1 * hessian_z1y1
                hessian_z1z2 = -1 * hessian_z1z1 
                
                hessian_x2x1 = hessian_x1x2
                hessian_x2y1 = hessian_y1x2
                hessian_x2z1 = hessian_z1x2
                hessian_x2x2 = -1 * hessian_x2x1
                hessian_x2y2 = -1 * hessian_x2y1
                hessian_x2z2 = -1 * hessian_x2z1

                hessian_y2x1 = hessian_x1y2
                hessian_y2y1 = hessian_y1y2
                hessian_y2z1 = hessian_z1y2
                hessian_y2x2 = -1 * hessian_y2x1
                hessian_y2y2 = -1 * hessian_y2y1
                hessian_y2z2 = -1 * hessian_y2z1
                
                hessian_z2x1 = hessian_x1z2
                hessian_z2y1 = hessian_y1z2
                hessian_z2z1 = hessian_z1z2
                hessian_z2x2 = -1 * hessian_z2x1
                hessian_z2y2 = -1 * hessian_z2y1
                hessian_z2z2 = -1 * hessian_z2z1


                tmp_hess[3*(i-1)+0][3*(i-1)+0] = copy.copy(hessian_x1x1)
                tmp_hess[3*(i-1)+0][3*(i-1)+1] = copy.copy(hessian_x1y1)
                tmp_hess[3*(i-1)+0][3*(i-1)+2] = copy.copy(hessian_x1z1)
                tmp_hess[3*(i-1)+0][3*(j-1)+0] = copy.copy(hessian_x1x2)
                tmp_hess[3*(i-1)+0][3*(j-1)+1] = copy.copy(hessian_x1y2)
                tmp_hess[3*(i-1)+0][3*(j-1)+2] = copy.copy(hessian_x1z2)
                
                tmp_hess[3*(i-1)+1][3*(i-1)+0] = copy.copy(hessian_y1x1)
                tmp_hess[3*(i-1)+1][3*(i-1)+1] = copy.copy(hessian_y1y1)
                tmp_hess[3*(i-1)+1][3*(i-1)+2] = copy.copy(hessian_y1z1)
                tmp_hess[3*(i-1)+1][3*(j-1)+0] = copy.copy(hessian_y1x2)
                tmp_hess[3*(i-1)+1][3*(j-1)+1] = copy.copy(hessian_y1y2)
                tmp_hess[3*(i-1)+1][3*(j-1)+2] = copy.copy(hessian_y1z2)
                
                tmp_hess[3*(i-1)+2][3*(i-1)+0] = copy.copy(hessian_z1x1)
                tmp_hess[3*(i-1)+2][3*(i-1)+1] = copy.copy(hessian_z1y1)
                tmp_hess[3*(i-1)+2][3*(i-1)+2] = copy.copy(hessian_z1z1)
                tmp_hess[3*(i-1)+2][3*(j-1)+0] = copy.copy(hessian_z1x2)
                tmp_hess[3*(i-1)+2][3*(j-1)+1] = copy.copy(hessian_z1y2)
                tmp_hess[3*(i-1)+2][3*(j-1)+2] = copy.copy(hessian_z1z2)
                
                tmp_hess[3*(j-1)+0][3*(i-1)+0] = copy.copy(hessian_x2x1)
                tmp_hess[3*(j-1)+0][3*(i-1)+1] = copy.copy(hessian_x2y1)
                tmp_hess[3*(j-1)+0][3*(i-1)+2] = copy.copy(hessian_x2z1)
                tmp_hess[3*(j-1)+0][3*(j-1)+0] = copy.copy(hessian_x2x2)
                tmp_hess[3*(j-1)+0][3*(j-1)+1] = copy.copy(hessian_x2y2)
                tmp_hess[3*(j-1)+0][3*(j-1)+2] = copy.copy(hessian_x2z2)
                
                tmp_hess[3*(j-1)+1][3*(i-1)+0] = copy.copy(hessian_y2x1)
                tmp_hess[3*(j-1)+1][3*(i-1)+1] = copy.copy(hessian_y2y1)
                tmp_hess[3*(j-1)+1][3*(i-1)+2] = copy.copy(hessian_y2z1)
                tmp_hess[3*(j-1)+1][3*(j-1)+0] = copy.copy(hessian_y2x2)
                tmp_hess[3*(j-1)+1][3*(j-1)+1] = copy.copy(hessian_y2y2)
                tmp_hess[3*(j-1)+1][3*(j-1)+2] = copy.copy(hessian_y2z2)
                
                tmp_hess[3*(j-1)+2][3*(i-1)+0] = copy.copy(hessian_z2x1)
                tmp_hess[3*(j-1)+2][3*(i-1)+1] = copy.copy(hessian_z2y1)
                tmp_hess[3*(j-1)+2][3*(i-1)+2] = copy.copy(hessian_z2z1)
                tmp_hess[3*(j-1)+2][3*(j-1)+0] = copy.copy(hessian_z2x2)
                tmp_hess[3*(j-1)+2][3*(j-1)+1] = copy.copy(hessian_z2y2)
                tmp_hess[3*(j-1)+2][3*(j-1)+2] = copy.copy(hessian_z2z2)
            
                hessian = hessian + tmp_hess
            
            return hessian
   
   

        def calc_AFIR_pot(geom_num_list, gamma, fragm_1, fragm_2, element_list):
            """
            ###  Reference  ###
             Chem. Rec., 2016, 16, 2232
             J. Comput. Chem., 2018, 39, 233
             WIREs Comput. Mol. Sci., 2021, 11, e1538
            """
            R_0 = 3.8164/self.bohr2angstroms #ang.→bohr
            EPSIRON = 1.0061/self.hartree2kjmol #kj/mol→hartree
            if gamma > 0.0 or gamma < 0.0:
                alpha = (gamma/self.hartree2kjmol) / ((2 ** (-1/6) - (1 + np.sqrt(1 + (abs(gamma/self.hartree2kjmol) / EPSIRON))) ** (-1/6))*R_0) #hartree/Bohr
            else:
                alpha = 0.0
            A = 0.0
            B = 0.0
            
            p = 6.0

            for i, j in itertools.product(fragm_1, fragm_2):
                R_i = covalent_radii_lib(element_list[i-1])
                R_j = covalent_radii_lib(element_list[j-1])
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                omega = ((R_i + R_j) / vector) ** p #no unit
                A += omega * vector
                B += omega
            
            energy = alpha*(A/B)#A/B:Bohr
            return energy #hartree
        
        def calc_AFIR_grad(geom_num_list, gamma, fragm_1, fragm_2, element_list):
            grad = geom_num_list*0.0
            R_0 = 3.8164/self.bohr2angstroms #ang.→bohr
            EPSIRON = 1.0061/self.hartree2kjmol #kj/mol→hartree
            
            if gamma > 0.0 or gamma < 0.0:
                alpha = (gamma/self.hartree2kjmol) / ((2 ** (-1/6) - (1 + np.sqrt(1 + (abs(gamma/self.hartree2kjmol) / EPSIRON))) ** (-1/6))*R_0) #hartree/Bohr
            else:
                alpha = 0.0
                
            A = 0.0
            B = 0.0
            p = 6.0

            for i, j in itertools.product(fragm_1, fragm_2):
                R_i = covalent_radii_lib(element_list[i-1])
                R_j = covalent_radii_lib(element_list[j-1])
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                omega = ((R_i + R_j) / vector) ** p #no unit
                A += omega * vector
                B += omega
                
            for i, j in itertools.product(fragm_1, fragm_2):
                R_i = covalent_radii_lib(element_list[i-1])
                R_j = covalent_radii_lib(element_list[j-1])
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                omega = ((R_i + R_j) / vector) ** p 
                
                C_x = (1 - p) * omega * ((geom_num_list[i-1][0] - geom_num_list[j-1][0]) / vector)     
                C_y = (1 - p) * omega * ((geom_num_list[i-1][1] - geom_num_list[j-1][1]) / vector)      
                C_z = (1 - p) * omega * ((geom_num_list[i-1][2] - geom_num_list[j-1][2]) / vector)
                D_x = -1 * p * omega * ((geom_num_list[i-1][0] - geom_num_list[j-1][0]) / vector ** 2)   
                D_y = -1 * p * omega * ((geom_num_list[i-1][1] - geom_num_list[j-1][1]) / vector ** 2)   
                D_z = -1 * p * omega * ((geom_num_list[i-1][2] - geom_num_list[j-1][2]) / vector ** 2)   
            
                grad[i-1] += alpha * np.array([(C_x/B) - ((D_x*A)/B ** 2), (C_y/B) - ((D_y*A)/B ** 2), (C_z/B) - ((D_z*A)/B ** 2)]  ,dtype="float64")
                grad[j-1] -= alpha * np.array([(C_x/B) - ((D_x*A)/B ** 2), (C_y/B) - ((D_y*A)/B ** 2), (C_z/B) - ((D_z*A)/B ** 2)]  ,dtype="float64")
                
            
            return grad
        
        
        def calc_AFIR_hess(geom_num_list, gamma, fragm_1, fragm_2, element_list, hessian):
            R_0 = 3.8164/self.bohr2angstroms #ang.→bohr
            EPSIRON = 1.0061/self.hartree2kjmol #kj/mol→hartree
            tmp_hess = (hessian*0.0).tolist()
            if gamma > 0.0 or gamma < 0.0:
                alpha = (gamma/self.hartree2kjmol) / ((2 ** (-1/6) - (1 + np.sqrt(1 + (abs(gamma/self.hartree2kjmol) / EPSIRON))) ** (-1/6))*R_0) #hartree/Bohr
            else:
                alpha = 0.0
                
            A = 0.0
            B = 0.0
            p = 6.0

            for i, j in itertools.product(fragm_1, fragm_2):
                R_i = covalent_radii_lib(element_list[i-1])
                R_j = covalent_radii_lib(element_list[j-1])
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                omega = ((R_i + R_j) / vector) ** p #no unit
                A += omega * vector
                B += omega
            
            for i, j in itertools.product(fragm_1, fragm_2):
                R_i = covalent_radii_lib(element_list[i-1])
                R_j = covalent_radii_lib(element_list[j-1])
                vector = np.linalg.norm(geom_num_list[i-1] - geom_num_list[j-1], ord=2) #bohr
                omega = ((R_i + R_j) / vector) ** p
             
                for k in fragm_2:
                
                    R_k = covalent_radii_lib(element_list[k-1])
                    vector_2 = np.linalg.norm(geom_num_list[i-1] - geom_num_list[k-1], ord=2) #bohr
                    omega_2 = ((R_i + R_k) / vector_2) ** p
                    
                    if j == k:
                        X = -1.0 * p * (omega_2 / vector_2) * (1.0 / B)
                        Y = A * (-1.0 * p) * (omega_2 / vector_2) 
                        Z = B ** 2
                        M = (1.0 - p) * (omega / B) + (A / B ** 2) * p * (omega / vector)
                    else:
                        X = 0.0
                        Y = 0.0
                        Z = 0.0
                        M = 0.0
                    L = (1.0 - p) * (X + (p / B ** 2) * omega * (omega_2 / vector_2)) + p * ((1.0 - p) * omega * omega_2 + Y) * (1.0 / ( B ** 2 * vector)) + (-1.0 / ( B ** 4 * vector ** 2)) * A * omega * (2.0 * B * vector * (-1.0 * p * (omega_2 / vector_2))) 
                
                    hessian_x1x1 = alpha * (geom_num_list[i-1][0] - geom_num_list[j-1][0] / vector) * (geom_num_list[i-1][0] - geom_num_list[k-1][0] / vector_2) * (L - (M / vector))
                    hessian_x1y1 = alpha * (geom_num_list[i-1][0] - geom_num_list[j-1][0] / vector) * (geom_num_list[i-1][1] - geom_num_list[k-1][1] / vector_2) * (L - (M / vector))
                    hessian_x1z1 = alpha * (geom_num_list[i-1][0] - geom_num_list[j-1][0] / vector) * (geom_num_list[i-1][2] - geom_num_list[k-1][2] / vector_2) * (L - (M / vector))
                    hessian_x1x2 = -1 * hessian_x1x1
                    hessian_x1y2 = -1 * hessian_x1y1
                    hessian_x1z2 = -1 * hessian_x1z1
                    
                    hessian_y1x1 = hessian_x1y1
                    hessian_y1y1 = alpha * (geom_num_list[i-1][1] - geom_num_list[j-1][1] / vector) * (geom_num_list[i-1][1] - geom_num_list[k-1][1] / vector_2) * (L - (M / vector))
                    hessian_y1z1 = alpha * (geom_num_list[i-1][1] - geom_num_list[j-1][1] / vector) * (geom_num_list[i-1][2] - geom_num_list[k-1][2] / vector_2) * (L - (M / vector))
                    hessian_y1x2 = -1 * hessian_y1x1
                    hessian_y1y2 = -1 * hessian_y1y1
                    hessian_y1z2 = -1 * hessian_y1z1
                    
                    hessian_z1x1 = hessian_x1z1
                    hessian_z1y1 = hessian_y1z1
                    hessian_z1z1 = alpha * (geom_num_list[i-1][2] - geom_num_list[j-1][2] / vector) * (geom_num_list[i-1][2] - geom_num_list[k-1][2] / vector_2) * (L - (M / vector))
                    hessian_z1x2 = -1 * hessian_z1x1
                    hessian_z1y2 = -1 * hessian_z1y1
                    hessian_z1z2 = -1 * hessian_z1z1 
                    
                    hessian_x2x1 = hessian_x1x2
                    hessian_x2y1 = hessian_y1x2
                    hessian_x2z1 = hessian_z1x2
                    hessian_x2x2 = -1 * hessian_x2x1
                    hessian_x2y2 = -1 * hessian_x2y1
                    hessian_x2z2 = -1 * hessian_x2z1

                    hessian_y2x1 = hessian_x1y2
                    hessian_y2y1 = hessian_y1y2
                    hessian_y2z1 = hessian_z1y2
                    hessian_y2x2 = -1 * hessian_y2x1
                    hessian_y2y2 = -1 * hessian_y2y1
                    hessian_y2z2 = -1 * hessian_y2z1
                    
                    hessian_z2x1 = hessian_x1z2
                    hessian_z2y1 = hessian_y1z2
                    hessian_z2z1 = hessian_z1z2
                    hessian_z2x2 = -1 * hessian_z2x1
                    hessian_z2y2 = -1 * hessian_z2y1
                    hessian_z2z2 = -1 * hessian_z2z1


                    tmp_hess[3*(i-1)+0][3*(i-1)+0] += hessian_x1x1
                    tmp_hess[3*(i-1)+0][3*(i-1)+1] += hessian_x1y1
                    tmp_hess[3*(i-1)+0][3*(i-1)+2] += hessian_x1z1
                    tmp_hess[3*(i-1)+0][3*(j-1)+0] += hessian_x1x2
                    tmp_hess[3*(i-1)+0][3*(j-1)+1] += hessian_x1y2
                    tmp_hess[3*(i-1)+0][3*(j-1)+2] += hessian_x1z2
                    
                    tmp_hess[3*(i-1)+1][3*(i-1)+0] += hessian_y1x1
                    tmp_hess[3*(i-1)+1][3*(i-1)+1] += hessian_y1y1
                    tmp_hess[3*(i-1)+1][3*(i-1)+2] += hessian_y1z1
                    tmp_hess[3*(i-1)+1][3*(j-1)+0] += hessian_y1x2
                    tmp_hess[3*(i-1)+1][3*(j-1)+1] += hessian_y1y2
                    tmp_hess[3*(i-1)+1][3*(j-1)+2] += hessian_y1z2
                    
                    tmp_hess[3*(i-1)+2][3*(i-1)+0] += hessian_z1x1
                    tmp_hess[3*(i-1)+2][3*(i-1)+1] += hessian_z1y1
                    tmp_hess[3*(i-1)+2][3*(i-1)+2] += hessian_z1z1
                    tmp_hess[3*(i-1)+2][3*(j-1)+0] += hessian_z1x2
                    tmp_hess[3*(i-1)+2][3*(j-1)+1] += hessian_z1y2
                    tmp_hess[3*(i-1)+2][3*(j-1)+2] += hessian_z1z2
                    
                    tmp_hess[3*(j-1)+0][3*(i-1)+0] += hessian_x2x1
                    tmp_hess[3*(j-1)+0][3*(i-1)+1] += hessian_x2y1
                    tmp_hess[3*(j-1)+0][3*(i-1)+2] += hessian_x2z1
                    tmp_hess[3*(j-1)+0][3*(j-1)+0] += hessian_x2x2
                    tmp_hess[3*(j-1)+0][3*(j-1)+1] += hessian_x2y2
                    tmp_hess[3*(j-1)+0][3*(j-1)+2] += hessian_x2z2
                    
                    tmp_hess[3*(j-1)+1][3*(i-1)+0] += hessian_y2x1
                    tmp_hess[3*(j-1)+1][3*(i-1)+1] += hessian_y2y1
                    tmp_hess[3*(j-1)+1][3*(i-1)+2] += hessian_y2z1
                    tmp_hess[3*(j-1)+1][3*(j-1)+0] += hessian_y2x2
                    tmp_hess[3*(j-1)+1][3*(j-1)+1] += hessian_y2y2
                    tmp_hess[3*(j-1)+1][3*(j-1)+2] += hessian_y2z2
                    
                    tmp_hess[3*(j-1)+2][3*(i-1)+0] += hessian_z2x1
                    tmp_hess[3*(j-1)+2][3*(i-1)+1] += hessian_z2y1
                    tmp_hess[3*(j-1)+2][3*(i-1)+2] += hessian_z2z1
                    tmp_hess[3*(j-1)+2][3*(j-1)+0] += hessian_z2x2
                    tmp_hess[3*(j-1)+2][3*(j-1)+1] += hessian_z2y2
                    tmp_hess[3*(j-1)+2][3*(j-1)+2] += hessian_z2z2
                     
            
            
            hessian = hessian + np.array(tmp_hess, dtype="float64")
            return hessian
              
        def calc_keep_potential(coord1, coord2, spring_const, keep_dist):
            vector = np.linalg.norm((coord1 - coord2), ord=2)
            energy = 0.5 * spring_const * (vector - keep_dist/self.bohr2angstroms) ** 2
            return energy #hartree
            
        def calc_keep_potential_grad(coord1, coord2, spring_const, keep_dist):
            vector = np.linalg.norm((coord1 - coord2), ord=2)
            grad_x_1 = spring_const * ((vector - keep_dist/self.bohr2angstroms) * (coord1[0] - coord2[0])) / (vector)
            grad_y_1 = spring_const * ((vector - keep_dist/self.bohr2angstroms) * (coord1[1] - coord2[1])) / (vector)
            grad_z_1 = spring_const * ((vector - keep_dist/self.bohr2angstroms) * (coord1[2] - coord2[2])) / (vector)

            grad_1 = np.array([grad_x_1, grad_y_1, grad_z_1], dtype="float64") #hartree/Bohr
            grad_2 = np.array([-1.0 * grad_x_1, -1.0 * grad_y_1, -1.0 * grad_z_1], dtype="float64") #hartree/Bohr
            return grad_1, grad_2 #hartree/Bohr
            

        def calc_keep_potential_hess(coord1, coord2, spring_const, keep_dist, coord1_num, coord2_num, hessian):
            
            vector = np.linalg.norm((coord1 - coord2), ord=2)
            tmp_hess = (hessian*0.0)
            hessian_x1x1 = (spring_const * (keep_dist/self.bohr2angstroms) / vector ** 3 ) * (coord1[0] - coord2[0]) ** 2 + spring_const * ((vector - (keep_dist/self.bohr2angstroms)) / vector)
            hessian_x1y1 = (spring_const * (keep_dist/self.bohr2angstroms) / vector ** 3 ) * (coord1[0] - coord2[0]) * (coord1[1] - coord2[1])
            hessian_x1z1 = (spring_const * (keep_dist/self.bohr2angstroms) / vector ** 3 ) * (coord1[0] - coord2[0]) * (coord1[2] - coord2[2])
            hessian_x1x2 = -1 * hessian_x1x1
            hessian_x1y2 = -1 * hessian_x1y1
            hessian_x1z2 = -1 * hessian_x1z1
            
            hessian_y1x1 = hessian_x1y1
            hessian_y1y1 = (spring_const * (keep_dist/self.bohr2angstroms) / vector ** 3 ) * (coord1[1] - coord2[1]) ** 2 + spring_const * ((vector - (keep_dist/self.bohr2angstroms)) / vector)
            hessian_y1z1 = (spring_const * (keep_dist/self.bohr2angstroms) / vector ** 3 ) * (coord1[1] - coord2[1]) * (coord1[2] - coord2[2])
            hessian_y1x2 = -1 * hessian_y1x1
            hessian_y1y2 = -1 * hessian_y1y1
            hessian_y1z2 = -1 * hessian_y1z1
            
            hessian_z1x1 = hessian_x1z1
            hessian_z1y1 = hessian_y1z1
            hessian_z1z1 = (spring_const * (keep_dist/self.bohr2angstroms) / vector ** 3 ) * (coord1[2] - coord2[2]) ** 2 + spring_const * ((vector - (keep_dist/self.bohr2angstroms)) / vector)
            hessian_z1x2 = -1 * hessian_z1x1
            hessian_z1y2 = -1 * hessian_z1y1
            hessian_z1z2 = -1 * hessian_z1z1 
            
            hessian_x2x1 = hessian_x1x2
            hessian_x2y1 = hessian_y1x2
            hessian_x2z1 = hessian_z1x2
            hessian_x2x2 = -1 * hessian_x2x1
            hessian_x2y2 = -1 * hessian_x2y1
            hessian_x2z2 = -1 * hessian_x2z1

            hessian_y2x1 = hessian_x1y2
            hessian_y2y1 = hessian_y1y2
            hessian_y2z1 = hessian_z1y2
            hessian_y2x2 = -1 * hessian_y2x1
            hessian_y2y2 = -1 * hessian_y2y1
            hessian_y2z2 = -1 * hessian_y2z1
            
            hessian_z2x1 = hessian_x1z2
            hessian_z2y1 = hessian_y1z2
            hessian_z2z1 = hessian_z1z2
            hessian_z2x2 = -1 * hessian_z2x1
            hessian_z2y2 = -1 * hessian_z2y1
            hessian_z2z2 = -1 * hessian_z2z1

            tmp_hess[3*(coord1_num-1)+0][3*(coord1_num-1)+0] = copy.copy(hessian_x1x1)
            tmp_hess[3*(coord1_num-1)+0][3*(coord1_num-1)+1] = copy.copy(hessian_x1y1)
            tmp_hess[3*(coord1_num-1)+0][3*(coord1_num-1)+2] = copy.copy(hessian_x1z1)
            tmp_hess[3*(coord1_num-1)+0][3*(coord2_num-1)+0] = copy.copy(hessian_x1x2)
            tmp_hess[3*(coord1_num-1)+0][3*(coord2_num-1)+1] = copy.copy(hessian_x1y2)
            tmp_hess[3*(coord1_num-1)+0][3*(coord2_num-1)+2] = copy.copy(hessian_x1z2)
            
            tmp_hess[3*(coord1_num-1)+1][3*(coord1_num-1)+0] = copy.copy(hessian_y1x1)
            tmp_hess[3*(coord1_num-1)+1][3*(coord1_num-1)+1] = copy.copy(hessian_y1y1)
            tmp_hess[3*(coord1_num-1)+1][3*(coord1_num-1)+2] = copy.copy(hessian_y1z1)
            tmp_hess[3*(coord1_num-1)+1][3*(coord2_num-1)+0] = copy.copy(hessian_y1x2)
            tmp_hess[3*(coord1_num-1)+1][3*(coord2_num-1)+1] = copy.copy(hessian_y1y2)
            tmp_hess[3*(coord1_num-1)+1][3*(coord2_num-1)+2] = copy.copy(hessian_y1z2)
            
            tmp_hess[3*(coord1_num-1)+2][3*(coord1_num-1)+0] = copy.copy(hessian_z1x1)
            tmp_hess[3*(coord1_num-1)+2][3*(coord1_num-1)+1] = copy.copy(hessian_z1y1)
            tmp_hess[3*(coord1_num-1)+2][3*(coord1_num-1)+2] = copy.copy(hessian_z1z1)
            tmp_hess[3*(coord1_num-1)+2][3*(coord2_num-1)+0] = copy.copy(hessian_z1x2)
            tmp_hess[3*(coord1_num-1)+2][3*(coord2_num-1)+1] = copy.copy(hessian_z1y2)
            tmp_hess[3*(coord1_num-1)+2][3*(coord2_num-1)+2] = copy.copy(hessian_z1z2)
            
            tmp_hess[3*(coord2_num-1)+0][3*(coord1_num-1)+0] = copy.copy(hessian_x2x1)
            tmp_hess[3*(coord2_num-1)+0][3*(coord1_num-1)+1] = copy.copy(hessian_x2y1)
            tmp_hess[3*(coord2_num-1)+0][3*(coord1_num-1)+2] = copy.copy(hessian_x2z1)
            tmp_hess[3*(coord2_num-1)+0][3*(coord2_num-1)+0] = copy.copy(hessian_x2x2)
            tmp_hess[3*(coord2_num-1)+0][3*(coord2_num-1)+1] = copy.copy(hessian_x2y2)
            tmp_hess[3*(coord2_num-1)+0][3*(coord2_num-1)+2] = copy.copy(hessian_x2z2)
            
            tmp_hess[3*(coord2_num-1)+1][3*(coord1_num-1)+0] = copy.copy(hessian_y2x1)
            tmp_hess[3*(coord2_num-1)+1][3*(coord1_num-1)+1] = copy.copy(hessian_y2y1)
            tmp_hess[3*(coord2_num-1)+1][3*(coord1_num-1)+2] = copy.copy(hessian_y2z1)
            tmp_hess[3*(coord2_num-1)+1][3*(coord2_num-1)+0] = copy.copy(hessian_y2x2)
            tmp_hess[3*(coord2_num-1)+1][3*(coord2_num-1)+1] = copy.copy(hessian_y2y2)
            tmp_hess[3*(coord2_num-1)+1][3*(coord2_num-1)+2] = copy.copy(hessian_y2z2)
            
            tmp_hess[3*(coord2_num-1)+2][3*(coord1_num-1)+0] = copy.copy(hessian_z2x1)
            tmp_hess[3*(coord2_num-1)+2][3*(coord1_num-1)+1] = copy.copy(hessian_z2y1)
            tmp_hess[3*(coord2_num-1)+2][3*(coord1_num-1)+2] = copy.copy(hessian_z2z1)
            tmp_hess[3*(coord2_num-1)+2][3*(coord2_num-1)+0] = copy.copy(hessian_z2x2)
            tmp_hess[3*(coord2_num-1)+2][3*(coord2_num-1)+1] = copy.copy(hessian_z2y2)
            tmp_hess[3*(coord2_num-1)+2][3*(coord2_num-1)+2] = copy.copy(hessian_z2z2)
            
            hessian = hessian + tmp_hess
            
            return hessian

          
        def calc_anharmonic_keep_potential(coord1, coord2, spring_const, keep_dist, pot_depth):
            vector = np.linalg.norm((coord1 - coord2), ord=2)
            if pot_depth != 0.0:
                energy = pot_depth * ( 1.0 - np.exp( - np.sqrt(spring_const / (2 * pot_depth)) * (vector - keep_dist/self.bohr2angstroms)) ) ** 2
            else:
                energy = 0.0
            return energy #hartree
            
        
        def calc_anharmonic_keep_potential_grad(coord1, coord2, spring_const, keep_dist, pot_depth):
            vector = np.linalg.norm((coord1 - coord2), ord=2)
            if pot_depth != 0.0:
                a = np.sqrt(spring_const / (2 * pot_depth))
                grad_x = 2 * a * pot_depth * ( 1.0 - np.exp( - a * (vector - keep_dist/self.bohr2angstroms)) ) * np.exp( - a * (vector - keep_dist/self.bohr2angstroms)) * ( (coord1[0] - coord2[0]) / vector )
                grad_y = 2 * a * pot_depth * ( 1.0 - np.exp( - a * (vector - keep_dist/self.bohr2angstroms)) ) * np.exp( - a * (vector - keep_dist/self.bohr2angstroms)) * ( (coord1[1] - coord2[1]) / vector )
                grad_z = 2 * a * pot_depth * ( 1.0 - np.exp( - a * (vector - keep_dist/self.bohr2angstroms)) ) * np.exp( - a * (vector - keep_dist/self.bohr2angstroms)) * ( (coord1[2] - coord2[2]) / vector )
                
                
                grad_1, grad_2 = np.array([grad_x, grad_y, grad_z], dtype="float64"), np.array([-1 * grad_x, -1 * grad_y, -1 * grad_z], dtype="float64")
            else:
                grad_1, grad_2 = np.array([0.0,0.0,0.0], dtype="float64"), np.array([0.0,0.0,0.0], dtype="float64")
                
            return grad_1, grad_2 #hartree/Bohr
            
        def calc_anharmonic_keep_potential_hess(coord1, coord2, spring_const, keep_dist, pot_depth, coord1_num, coord2_num, hessian):
            vector = np.linalg.norm((coord1 - coord2), ord=2)
            if pot_depth != 0.0:
                tmp_hess = hessian*0.0
                a = np.sqrt(spring_const / (2 * pot_depth))
                
                hessian_x1x1 = 2.0 * pot_depth * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) * (((a * (coord1[0] - coord2[0]))/ vector ) ** 2 * (2 * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) - 1.0) + (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) * a * ((1.0 / vector) - ((coord1[0] - coord2[0]) ** 2/vector ** 3)))
                hessian_x1y1 = 2.0 * pot_depth * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) * (coord1[0] - coord2[0]) * (coord1[1] - coord2[1]) * (a / vector) * ((a / vector) * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) - (a / vector) * (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) - (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) * (1.0 / vector ** 2))
                hessian_x1z1 = 2.0 * pot_depth * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) * (coord1[0] - coord2[0]) * (coord1[2] - coord2[2]) * (a / vector) * ((a / vector) * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) - (a / vector) * (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) - (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) * (1.0 / vector ** 2))
                hessian_x1x2 = -1 * hessian_x1x1
                hessian_x1y2 = -1 * hessian_x1y1
                hessian_x1z2 = -1 * hessian_x1z1
                
                hessian_y1x1 = hessian_x1y1
                hessian_y1y1 = 2.0 * pot_depth * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) * (((a * (coord1[1] - coord2[1]))/ vector ) ** 2 * (2 * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) - 1.0) + (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) * a * ((1.0 / vector) - ((coord1[1] - coord2[1]) ** 2/vector ** 3)))
                hessian_y1z1 = 2.0 * pot_depth * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) * (coord1[2] - coord2[2]) * (coord1[1] - coord2[1]) * (a / vector) * ((a / vector) * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) - (a / vector) * (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) - (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) * (1.0 / vector ** 2))
                hessian_y1x2 = -1 * hessian_y1x1
                hessian_y1y2 = -1 * hessian_y1y1
                hessian_y1z2 = -1 * hessian_y1z1
                
                hessian_z1x1 = hessian_x1z1
                hessian_z1y1 = hessian_y1z1
                hessian_z1z1 = 2.0 * pot_depth * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) * (((a * (coord1[2] - coord2[2]))/ vector ) ** 2 * (2 * np.exp(- a * (vector - keep_dist/self.bohr2angstroms)) - 1.0) + (1.0 - np.exp(- a * (vector - keep_dist/self.bohr2angstroms))) * a * ((1.0 / vector) - ((coord1[2] - coord2[2]) ** 2/vector ** 3)))
                hessian_z1x2 = -1 * hessian_z1x1
                hessian_z1y2 = -1 * hessian_z1y1
                hessian_z1z2 = -1 * hessian_z1z1 
                
                hessian_x2x1 = hessian_x1x2
                hessian_x2y1 = hessian_y1x2
                hessian_x2z1 = hessian_z1x2
                hessian_x2x2 = -1 * hessian_x2x1
                hessian_x2y2 = -1 * hessian_x2y1
                hessian_x2z2 = -1 * hessian_x2z1

                hessian_y2x1 = hessian_x1y2
                hessian_y2y1 = hessian_y1y2
                hessian_y2z1 = hessian_z1y2
                hessian_y2x2 = -1 * hessian_y2x1
                hessian_y2y2 = -1 * hessian_y2y1
                hessian_y2z2 = -1 * hessian_y2z1
                
                hessian_z2x1 = hessian_x1z2
                hessian_z2y1 = hessian_y1z2
                hessian_z2z1 = hessian_z1z2
                hessian_z2x2 = -1 * hessian_z2x1
                hessian_z2y2 = -1 * hessian_z2y1
                hessian_z2z2 = -1 * hessian_z2z1

                tmp_hess[3*(coord1_num-1)+0][3*(coord1_num-1)+0] = copy.copy(hessian_x1x1)
                tmp_hess[3*(coord1_num-1)+0][3*(coord1_num-1)+1] = copy.copy(hessian_x1y1)
                tmp_hess[3*(coord1_num-1)+0][3*(coord1_num-1)+2] = copy.copy(hessian_x1z1)
                tmp_hess[3*(coord1_num-1)+0][3*(coord2_num-1)+0] = copy.copy(hessian_x1x2)
                tmp_hess[3*(coord1_num-1)+0][3*(coord2_num-1)+1] = copy.copy(hessian_x1y2)
                tmp_hess[3*(coord1_num-1)+0][3*(coord2_num-1)+2] = copy.copy(hessian_x1z2)
                
                tmp_hess[3*(coord1_num-1)+1][3*(coord1_num-1)+0] = copy.copy(hessian_y1x1)
                tmp_hess[3*(coord1_num-1)+1][3*(coord1_num-1)+1] = copy.copy(hessian_y1y1)
                tmp_hess[3*(coord1_num-1)+1][3*(coord1_num-1)+2] = copy.copy(hessian_y1z1)
                tmp_hess[3*(coord1_num-1)+1][3*(coord2_num-1)+0] = copy.copy(hessian_y1x2)
                tmp_hess[3*(coord1_num-1)+1][3*(coord2_num-1)+1] = copy.copy(hessian_y1y2)
                tmp_hess[3*(coord1_num-1)+1][3*(coord2_num-1)+2] = copy.copy(hessian_y1z2)
                
                tmp_hess[3*(coord1_num-1)+2][3*(coord1_num-1)+0] = copy.copy(hessian_z1x1)
                tmp_hess[3*(coord1_num-1)+2][3*(coord1_num-1)+1] = copy.copy(hessian_z1y1)
                tmp_hess[3*(coord1_num-1)+2][3*(coord1_num-1)+2] = copy.copy(hessian_z1z1)
                tmp_hess[3*(coord1_num-1)+2][3*(coord2_num-1)+0] = copy.copy(hessian_z1x2)
                tmp_hess[3*(coord1_num-1)+2][3*(coord2_num-1)+1] = copy.copy(hessian_z1y2)
                tmp_hess[3*(coord1_num-1)+2][3*(coord2_num-1)+2] = copy.copy(hessian_z1z2)
                
                tmp_hess[3*(coord2_num-1)+0][3*(coord1_num-1)+0] = copy.copy(hessian_x2x1)
                tmp_hess[3*(coord2_num-1)+0][3*(coord1_num-1)+1] = copy.copy(hessian_x2y1)
                tmp_hess[3*(coord2_num-1)+0][3*(coord1_num-1)+2] = copy.copy(hessian_x2z1)
                tmp_hess[3*(coord2_num-1)+0][3*(coord2_num-1)+0] = copy.copy(hessian_x2x2)
                tmp_hess[3*(coord2_num-1)+0][3*(coord2_num-1)+1] = copy.copy(hessian_x2y2)
                tmp_hess[3*(coord2_num-1)+0][3*(coord2_num-1)+2] = copy.copy(hessian_x2z2)
                
                tmp_hess[3*(coord2_num-1)+1][3*(coord1_num-1)+0] = copy.copy(hessian_y2x1)
                tmp_hess[3*(coord2_num-1)+1][3*(coord1_num-1)+1] = copy.copy(hessian_y2y1)
                tmp_hess[3*(coord2_num-1)+1][3*(coord1_num-1)+2] = copy.copy(hessian_y2z1)
                tmp_hess[3*(coord2_num-1)+1][3*(coord2_num-1)+0] = copy.copy(hessian_y2x2)
                tmp_hess[3*(coord2_num-1)+1][3*(coord2_num-1)+1] = copy.copy(hessian_y2y2)
                tmp_hess[3*(coord2_num-1)+1][3*(coord2_num-1)+2] = copy.copy(hessian_y2z2)
                
                tmp_hess[3*(coord2_num-1)+2][3*(coord1_num-1)+0] = copy.copy(hessian_z2x1)
                tmp_hess[3*(coord2_num-1)+2][3*(coord1_num-1)+1] = copy.copy(hessian_z2y1)
                tmp_hess[3*(coord2_num-1)+2][3*(coord1_num-1)+2] = copy.copy(hessian_z2z1)
                tmp_hess[3*(coord2_num-1)+2][3*(coord2_num-1)+0] = copy.copy(hessian_z2x2)
                tmp_hess[3*(coord2_num-1)+2][3*(coord2_num-1)+1] = copy.copy(hessian_z2y2)
                tmp_hess[3*(coord2_num-1)+2][3*(coord2_num-1)+2] = copy.copy(hessian_z2z2)
                
                hessian = hessian + tmp_hess
                
            else:
                pass
                
            return hessian
        
        def calc_keep_angle(coord1, coord2, coord3, spring_const, keep_angle):
            vector1 = coord1 - coord2
            vector2 = coord3 - coord2
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            dot_product = np.dot(vector1, vector2)
            cos_theta = dot_product / (magnitude1 * magnitude2)
            theta = np.arccos(cos_theta)
            energy = 0.5 * spring_const * (theta - np.radians(keep_angle)) ** 2
            return energy #hartree
        
        def calc_keep_angle_grad(coord1, coord2, coord3, spring_const, keep_angle):
            
            vector1 = coord1 - coord2
            vector2 = coord3 - coord2
            dot_product = np.dot(vector1, vector2)

            diff_for_x = vector1[1]*vector2[1] + vector1[2]*vector2[2]
            diff_for_y = vector1[0]*vector2[0] + vector1[2]*vector2[2]
            diff_for_z = vector1[1]*vector2[1] + vector1[0]*vector2[0]
            
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            cos_theta = dot_product / (magnitude1 * magnitude2)
            theta = np.arccos(cos_theta)
            dV_dtheta = spring_const * (theta - np.radians(keep_angle))
            A = np.sqrt(abs(1 - ( np.linalg.norm( vector1 * vector2, ord=2 ) ** 2 / ((magnitude1 ** 2 ) * (magnitude2 ** 2 )) )))
            
            D = magnitude2 * magnitude1 ** 3 * A + 1e-09 #Denominator should not be zero.
            E = magnitude1 * magnitude2 ** 3 * A + 1e-09 #Denominator should not be zero.
            
            grad_1_x = dV_dtheta * (vector1[0] * diff_for_x - vector2[0] * (vector1[1] ** 2 + vector1[2] ** 2)) / D
            grad_1_y = dV_dtheta * (vector1[1] * diff_for_y - vector2[1] * (vector1[2] ** 2 + vector1[0] ** 2)) / D
            grad_1_z = dV_dtheta * (vector1[2] * diff_for_z - vector2[2] * (vector1[1] ** 2 + vector1[0] ** 2)) / D
            grad_3_x = dV_dtheta * (vector2[0] * diff_for_x - vector1[0] * (vector2[1] ** 2 + vector2[2] ** 2)) / E
            grad_3_y = dV_dtheta * (vector2[1] * diff_for_y - vector1[1] * (vector2[0] ** 2 + vector2[2] ** 2)) / E
            grad_3_z = dV_dtheta * (vector2[2] * diff_for_z - vector1[2] * (vector2[1] ** 2 + vector2[0] ** 2)) / E

            
            grad_1 = np.array([grad_1_x, grad_1_y, grad_1_z] ,dtype="float64")
            grad_3 = np.array([grad_3_x, grad_3_y, grad_3_z] ,dtype="float64")
            grad_2 = - grad_1 - grad_3

            return grad_1, grad_2, grad_3 #hartree/Bohr

        
        def calc_keep_dihedral_angle(coord1, coord2, coord3, coord4, spring_const, keep_dihedral_angle):

            a1 = coord2 - coord1
            a2 = coord3 - coord2
            a3 = coord4 - coord3

            v1 = np.cross(a1, a2)
            v1 = v1 / np.linalg.norm(v1, ord=2)
            v2 = np.cross(a2, a3)
            v2 = v2 / np.linalg.norm(v2, ord=2)
            porm = np.sign((v1 * a3).sum(-1))
            angle = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)
            if not porm == 0:
                angle = angle * porm
            
            energy = 0.5 * spring_const * (angle - np.radians(keep_dihedral_angle)) ** 2
            
            return energy #hartree    
            

        def calc_keep_dihedral_angle_grad(coord1, coord2, coord3, coord4, spring_const, keep_dihedral_angle):

            a1 = coord2 - coord1
            a2 = coord3 - coord2
            a3 = coord4 - coord3

            v1 = np.cross(a1, a2)
            v1_hat = v1 / (v1 * v1).sum(-1)**0.5
            v2 = np.cross(a2, a3)
            v2_hat = v2 / (v2 * v2).sum(-1)**0.5

            yc_zb = a1[1]*a2[2] - a1[2]*a2[1]
            za_xc = a1[2]*a2[0] - a1[0]*a2[2]
            xb_ya = a1[0]*a2[1] - a1[1]*a2[0]
            
            bn_cm = a2[1]*a3[2] - a2[2]*a3[1]
            cl_an = a2[2]*a3[0] - a2[0]*a3[2]
            am_bl = a2[0]*a3[1] - a2[1]*a3[0]
            
            
            alpha = a2[0] ** 2 + a2[1] ** 2 + a2[2] ** 2
            beta = a1[1]*cl_an + a1[2]*am_bl + a1[0]*bn_cm
            gamma = (v1_hat*v2_hat).sum(-1)
            A = (v2_hat**2).sum(-1) 
            B = (v1_hat**2).sum(-1) 
            
            omega = gamma/((A ** 0.5)*(B ** 0.5))
            
            porm = np.sign((v1_hat * a3).sum(-1))
            
            angle = np.arccos(omega)
            
            if not porm == 0:
                angle = angle * porm

            dV = spring_const * (angle - np.radians(keep_dihedral_angle))
            d_omega = -1/np.sqrt(abs(1 - omega ** 2) + 1e-8) 
            
            tmp_denominator = (A ** 0.5) * (B ** 1.5)
            grad_i_x = dV * d_omega * ((-1 * alpha * yc_zb * beta)/tmp_denominator)
            grad_i_y = dV * d_omega * ((-1 * alpha * za_xc * beta)/tmp_denominator)
            grad_i_z = dV * d_omega * ((-1 * alpha * xb_ya * beta)/tmp_denominator)
            
            tmp_denominator = [(A ** 0.5) * (B ** 0.5), (A ** 1.5) * (B ** 0.5), (A ** 0.5) * (B ** 1.5)]
            grad_j_x = dV * d_omega * (((-1 * a3[2] * za_xc + a3[1] * xb_ya + a1[2] * cl_an -1 * a1[1] * am_bl)/tmp_denominator[0]) - (((a3[1] * am_bl -a3[2] * cl_an ) * gamma)/tmp_denominator[1]) - (((a1[2] * za_xc -a1[1]*xb_ya) * gamma)/tmp_denominator[2]))
            grad_j_y = dV * d_omega * ((( a3[2] * yc_zb - a3[0] * xb_ya - a1[2] * bn_cm + a1[0] * am_bl)/tmp_denominator[0]) - (((a3[2] * bn_cm -a3[0] * am_bl ) * gamma)/tmp_denominator[1]) - (((a1[0] * xb_ya -a1[2]*yc_zb) * gamma)/tmp_denominator[2]))
            grad_j_z = dV * d_omega * ((( -1*a3[1] * yc_zb + a3[0] * za_xc + a1[1] * bn_cm -1 * a1[0] * cl_an)/tmp_denominator[0]) - (((a3[0] * cl_an -a3[1] * bn_cm ) * gamma)/tmp_denominator[1]) - (((a1[1] * yc_zb -a1[0] * za_xc) * gamma)/tmp_denominator[2]))
            
            tmp_denominator = (B ** 0.5) * (A ** 1.5)
            grad_k_x = dV * d_omega * ( (alpha * bn_cm * (-1 * yc_zb * a3[0] + (a3[2]*a1[1] - a3[1]*a1[2]) * a2[0] + (a2[2]*a3[1] - a2[1]*a3[2]) * a1[0])) / tmp_denominator)
            grad_k_y = dV * d_omega * ( (-1 * alpha * cl_an * (-1 * za_xc * a3[1] + (a2[0]*a3[2] - a2[2]*a3[0] ) * a2[1] + (a3[0]*a1[2] - a3[2]*a1[0]) * a1[1] )) / tmp_denominator)
            grad_k_z = dV * d_omega * ( (alpha * am_bl * (-1 * xb_ya * a3[2] + (a2[1]*a3[0] - a2[0]*a3[1]) * a1[2] + (a3[0]*a1[2] - a3[0]*a1[2]) * a2[2])) / tmp_denominator)
            
            grad_i = np.array([grad_i_x, grad_i_y, grad_i_z] , dtype="float64")
            grad_j = np.array([grad_j_x, grad_j_y, grad_j_z] , dtype="float64")
            grad_k = np.array([grad_k_x, grad_k_y, grad_k_z] , dtype="float64")
            
            grad_1 = - grad_i
            grad_2 = grad_i - grad_j
            grad_3 = grad_j - grad_k
            grad_4 = grad_k
            
            return grad_1, grad_2, grad_3, grad_4 #hartree/bohr

            
        def calc_void_point_pot(coord, void_point_coord, spring_const, keep_dist, order):
            vector = np.linalg.norm((coord - void_point_coord), ord=2)
            energy = (1 / order) * spring_const * (vector - keep_dist/self.bohr2angstroms) ** order
            return energy #hartree
        
        def calc_void_point_pot_grad(coord, void_point_coord, spring_const, keep_dist, order):
        
            vector = np.linalg.norm((coord - void_point_coord), ord=2)
            grad_x = spring_const * ((vector - keep_dist/self.bohr2angstroms) ** (order - 1) / vector ) * (coord[0] - void_point_coord[0])
            grad_y = spring_const * ((vector - keep_dist/self.bohr2angstroms) ** (order - 1) / vector ) * (coord[1] - void_point_coord[1])
            grad_z = spring_const * ((vector - keep_dist/self.bohr2angstroms) ** (order - 1) / vector ) * (coord[2] - void_point_coord[2])
            grad = np.array([grad_x, grad_y, grad_z], dtype="float64")
            
            return grad #hartree/Bohr
        
        def calc_void_point_pot_hess(coord, void_point_coord, spring_const, keep_dist, order, coord_num, hessian):
            vector = np.linalg.norm((coord - void_point_coord), ord=2)
            tmp_hess = hessian*0.0
            hessian_xx = spring_const * (coord[0] - void_point_coord[0]) * ((((vector - keep_dist/self.bohr2angstroms) ** (order - 2) * (coord[0] - void_point_coord[0])) / vector ** 2) * ((order - 1) - (vector - keep_dist/self.bohr2angstroms))) + spring_const * ((vector - keep_dist/self.bohr2angstroms) ** (order - 1) / vector)
            hessian_xy = spring_const * (coord[0] - void_point_coord[0]) * ((((vector - keep_dist/self.bohr2angstroms) ** (order - 2) * (coord[1] - void_point_coord[1])) / vector ** 2) * ((order - 1) - (vector - keep_dist/self.bohr2angstroms)))
            hessian_xz = spring_const * (coord[0] - void_point_coord[0]) * ((((vector - keep_dist/self.bohr2angstroms) ** (order - 2) * (coord[2] - void_point_coord[2])) / vector ** 2) * ((order - 1) - (vector - keep_dist/self.bohr2angstroms)))
            
            hessian_yx = hessian_xy
            hessian_yy = spring_const * (coord[1] - void_point_coord[1]) * ((((vector - keep_dist/self.bohr2angstroms) ** (order - 2) * (coord[1] - void_point_coord[1])) / vector ** 2) * ((order - 1) - (vector - keep_dist/self.bohr2angstroms))) + spring_const * ((vector - keep_dist/self.bohr2angstroms) ** (order - 1) / vector)
            hessian_yz = spring_const * (coord[1] - void_point_coord[1]) * ((((vector - keep_dist/self.bohr2angstroms) ** (order - 2) * (coord[2] - void_point_coord[2])) / vector ** 2) * ((order - 1) - (vector - keep_dist/self.bohr2angstroms)))
            
            hessian_zx = hessian_xz
            hessian_zy = hessian_yz
            hessian_zz = spring_const * (coord[2] - void_point_coord[2]) * ((((vector - keep_dist/self.bohr2angstroms) ** (order - 2) * (coord[2] - void_point_coord[2])) / vector ** 2) * ((order - 1) - (vector - keep_dist/self.bohr2angstroms))) + spring_const * ((vector - keep_dist/self.bohr2angstroms) ** (order - 1) / vector)
            
            tmp_hess[3*coord_num+0][3*coord_num+0] = copy.copy(hessian_xx)
            tmp_hess[3*coord_num+0][3*coord_num+1] = copy.copy(hessian_xy)
            tmp_hess[3*coord_num+0][3*coord_num+2] = copy.copy(hessian_xz)
        
            tmp_hess[3*coord_num+1][3*coord_num+0] = copy.copy(hessian_yx)
            tmp_hess[3*coord_num+1][3*coord_num+1] = copy.copy(hessian_yy)
            tmp_hess[3*coord_num+1][3*coord_num+2] = copy.copy(hessian_yz)

            tmp_hess[3*coord_num+2][3*coord_num+0] = copy.copy(hessian_zx)
            tmp_hess[3*coord_num+2][3*coord_num+1] = copy.copy(hessian_zy)
            tmp_hess[3*coord_num+2][3*coord_num+2] = copy.copy(hessian_zz)
        
            hessian = hessian + tmp_hess
            
            return hessian
        
        def calc_gaussian_pot(geom_num_list, gau_pot_energy, initial_geom_num_list):#This function is just for fun. Thus, it is no scientific basis.
            geom_mean_coord = np.mean(geom_num_list, axis=0)
            A = gau_pot_energy/(self.hartree2kjmol * len(geom_num_list))
            energy = A*np.sum(np.exp(-(geom_num_list - initial_geom_num_list - geom_mean_coord) ** 2))
            
            return energy
        
        def calc_gaussian_pot_grad(geom_num_list, gau_pot_energy, initial_geom_num_list):#This function is just for fun. Thus, it is no scientific basis.
            A = gau_pot_energy/(self.hartree2kjmol * len(geom_num_list))
            geom_mean_coord = np.mean(geom_num_list, axis=0)
            grad = []
            for i in range(len(geom_num_list)):
                
                grad_x = -2.0 * A * (geom_num_list[i][0] - initial_geom_num_list[i][0] - geom_mean_coord[0]) * np.exp(-(geom_num_list[i][0] - initial_geom_num_list[i][0] - geom_mean_coord[0]) ** 2)
                grad_y = -2.0 * A * (geom_num_list[i][1] - initial_geom_num_list[i][1] - geom_mean_coord[1]) * np.exp(-(geom_num_list[i][1] - initial_geom_num_list[i][1] - geom_mean_coord[1]) ** 2)
                grad_z = -2.0 * A * (geom_num_list[i][2] - initial_geom_num_list[i][2] - geom_mean_coord[2]) * np.exp(-(geom_num_list[i][2] - initial_geom_num_list[i][2] - geom_mean_coord[2]) ** 2)
                
                grad.append(np.array([grad_x, grad_y, grad_z], dtype="float64"))
            
            return grad
            
        
                
        def calc_LJ_Repulsive_pot_v2(geom_num_list, well_scale , dist_scale, length, const_rep, const_attr, order_rep, order_attr, center, target, element_list):  
            energy = 0.0
            
            LJ_pot_center = geom_num_list[center[1]-1] + (length/self.bohr2angstroms) * (geom_num_list[center[1]-1] - geom_num_list[center[0]-1] / np.linalg.norm(geom_num_list[center[1]-1] - geom_num_list[center[0]-1])) 
            for i in target:
                UFF_VDW_well_depth = np.sqrt(well_scale*UFF_VDW_well_depth_lib(element_list[center[1]-1]) + well_scale*UFF_VDW_well_depth_lib(element_list[i-1]))
                UFF_VDW_distance = np.sqrt(UFF_VDW_distance_lib(element_list[center[1]-1])*dist_scale + UFF_VDW_distance_lib(element_list[i-1])*dist_scale)
                
                vector = np.linalg.norm(geom_num_list[i-1] - LJ_pot_center, ord=2) #bohr
                energy += UFF_VDW_well_depth * ( abs(const_rep) * ( UFF_VDW_distance / vector ) ** order_rep -1 * abs(const_attr) * ( UFF_VDW_distance / vector ) ** order_attr)
                
            return energy
        
        def calc_LJ_Repulsive_pot_v2_grad(geom_num_list, well_scale , dist_scale, length, const_rep, const_attr, order_rep, order_attr, center, target, element_list):
            grad = geom_num_list*0.0
            
            LJ_pot_center = geom_num_list[center[1]-1] + (length/self.bohr2angstroms) * (geom_num_list[center[1]-1] - geom_num_list[center[0]-1] / np.linalg.norm(geom_num_list[center[1]-1] - geom_num_list[center[0]-1]))
            
            for i in target:
                UFF_VDW_well_depth = np.sqrt(well_scale*UFF_VDW_well_depth_lib(element_list[center[1]-1]) + well_scale*UFF_VDW_well_depth_lib(element_list[i-1]))
                UFF_VDW_distance = np.sqrt(UFF_VDW_distance_lib(element_list[center[1]-1])*dist_scale + UFF_VDW_distance_lib(element_list[i-1])*dist_scale)
                
                vector = np.linalg.norm(geom_num_list[i-1] - LJ_pot_center, ord=2) #bohr
                grad_x_1 = UFF_VDW_well_depth * (( -1 * abs(const_rep) * order_rep * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 2) ) + abs(const_attr) * (order_attr) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 2) ) ) * (geom_num_list[i-1][0] - LJ_pot_center[0])) 
                grad_y_1 = UFF_VDW_well_depth * (( -1 * abs(const_rep) * order_rep * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 2) ) + abs(const_attr) * (order_attr) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 2) ) ) * (geom_num_list[i-1][1] - LJ_pot_center[1])) 
                grad_z_1 = UFF_VDW_well_depth * (( -1 * abs(const_rep) * order_rep * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 2) ) + abs(const_attr) * (order_attr) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 2) ) ) * (geom_num_list[i-1][2] - LJ_pot_center[2])) 

                grad[i-1] += np.array([grad_x_1,grad_y_1,grad_z_1], dtype="float64") #hartree/Bohr
                grad[center[1]-1-1] += np.array([-1.0 * grad_x_1, -1.0 * grad_y_1, -1.0 * grad_z_1], dtype="float64") #hartree/Bohr
                
            return grad
            
        def calc_LJ_Repulsive_pot_v2_hess(geom_num_list, well_scale , dist_scale, length, const_rep, const_attr, order_rep, order_attr, center, target, element_list, hessian):
            
            LJ_pot_center = geom_num_list[center[1]-1] + (length/self.bohr2angstroms) * (geom_num_list[center[1]-1] - geom_num_list[center[0]-1] / np.linalg.norm(geom_num_list[center[1]-1] - geom_num_list[center[0]-1]))
            
            for i in target:
                UFF_VDW_well_depth = np.sqrt(well_scale*UFF_VDW_well_depth_lib(element_list[center[1]-1]) + well_scale*UFF_VDW_well_depth_lib(element_list[i-1]))
                UFF_VDW_distance = np.sqrt(UFF_VDW_distance_lib(element_list[center[1]-1])*dist_scale + UFF_VDW_distance_lib(element_list[i-1])*dist_scale)
                
                vector = np.linalg.norm(geom_num_list[i-1] - LJ_pot_center, ord=2) #bohr
                
                hessian_x1x1 = UFF_VDW_well_depth * (( abs(const_rep) * order_rep * (order_rep + 2) * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 4) ) -1 * abs(const_attr) * (order_attr) * (order_attr + 2) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 4) ) ) * (geom_num_list[i-1][0] - LJ_pot_center[0])) * (geom_num_list[i-1][0] - LJ_pot_center[0]) + UFF_VDW_well_depth * (( -1 * abs(const_rep) * order_rep * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 2) ) + abs(const_attr) * (order_attr) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 2) ) ) * (geom_num_list[i-1][0] - LJ_pot_center[0])) 
                hessian_x1y1 = UFF_VDW_well_depth * (( abs(const_rep) * order_rep * (order_rep + 2) * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 4) ) -1 * abs(const_attr) * (order_attr) * (order_attr + 2) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 4) ) ) * (geom_num_list[i-1][0] - LJ_pot_center[0])) * (geom_num_list[i-1][1] - LJ_pot_center[1])
                hessian_x1z1 = UFF_VDW_well_depth * (( abs(const_rep) * order_rep * (order_rep + 2) * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 4) ) -1 * abs(const_attr) * (order_attr) * (order_attr + 2) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 4) ) ) * (geom_num_list[i-1][0] - LJ_pot_center[0])) * (geom_num_list[i-1][2] - LJ_pot_center[2])
                hessian_x1x2 = -1 * hessian_x1x1
                hessian_x1y2 = -1 * hessian_x1y1
                hessian_x1z2 = -1 * hessian_x1z1
                
                hessian_y1x1 = hessian_x1y1
                hessian_y1y1 = UFF_VDW_well_depth * (( abs(const_rep) * order_rep * (order_rep + 2) * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 4) ) -1 * abs(const_attr) * (order_attr) * (order_attr + 2) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 4) ) ) * (geom_num_list[i-1][1] - LJ_pot_center[1])) * (geom_num_list[i-1][1] - LJ_pot_center[1]) + UFF_VDW_well_depth * (( -1 * abs(const_rep) * order_rep * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 2) ) + abs(const_attr) * (order_attr) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 2) ) ) * (geom_num_list[i-1][1] - LJ_pot_center[1])) 
                hessian_y1z1 = 12 * UFF_VDW_well_depth * ((geom_num_list[i-1][1] - geom_num_list[j-1][1]) * (geom_num_list[i-1][2] - geom_num_list[j-1][2]) * (-8 * (UFF_VDW_distance ** 6 / vector ** 10) + 14 * (UFF_VDW_distance ** 12 / vector ** 16)))
                hessian_y1x2 = -1 * hessian_y1x1
                hessian_y1y2 = -1 * hessian_y1y1
                hessian_y1z2 = -1 * hessian_y1z1
                
                hessian_z1x1 = hessian_x1z1
                hessian_z1y1 = hessian_y1z1
                hessian_z1z1 = 12 * UFF_VDW_well_depth * (( abs(const_rep) * order_rep * (order_rep + 2) * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 4) ) -1 * abs(const_attr) * (order_attr) * (order_attr + 2) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 4) ) ) * (geom_num_list[i-1][2] - LJ_pot_center[2])) * (geom_num_list[i-1][2] - LJ_pot_center[2]) + UFF_VDW_well_depth * (( -1 * abs(const_rep) * order_rep * ( UFF_VDW_distance ** order_rep / vector ** (order_rep + 2) ) + abs(const_attr) * (order_attr) * ( UFF_VDW_distance ** (order_attr) / vector ** (order_attr + 2) ) ) * (geom_num_list[i-1][2] - LJ_pot_center[2])) 
                hessian_z1x2 = -1 * hessian_z1x1
                hessian_z1y2 = -1 * hessian_z1y1
                hessian_z1z2 = -1 * hessian_z1z1 
                
                hessian_x2x1 = hessian_x1x2
                hessian_x2y1 = hessian_y1x2
                hessian_x2z1 = hessian_z1x2
                hessian_x2x2 = -1 * hessian_x2x1
                hessian_x2y2 = -1 * hessian_x2y1
                hessian_x2z2 = -1 * hessian_x2z1

                hessian_y2x1 = hessian_x1y2
                hessian_y2y1 = hessian_y1y2
                hessian_y2z1 = hessian_z1y2
                hessian_y2x2 = -1 * hessian_y2x1
                hessian_y2y2 = -1 * hessian_y2y1
                hessian_y2z2 = -1 * hessian_y2z1
                
                hessian_z2x1 = hessian_x1z2
                hessian_z2y1 = hessian_y1z2
                hessian_z2z1 = hessian_z1z2
                hessian_z2x2 = -1 * hessian_z2x1
                hessian_z2y2 = -1 * hessian_z2y1
                hessian_z2z2 = -1 * hessian_z2z1


                tmp_hess[3*(i-1)+0][3*(i-1)+0] = copy.copy(hessian_x1x1)
                tmp_hess[3*(i-1)+0][3*(i-1)+1] = copy.copy(hessian_x1y1)
                tmp_hess[3*(i-1)+0][3*(i-1)+2] = copy.copy(hessian_x1z1)
                tmp_hess[3*(i-1)+0][3*(j-1)+0] = copy.copy(hessian_x1x2)
                tmp_hess[3*(i-1)+0][3*(j-1)+1] = copy.copy(hessian_x1y2)
                tmp_hess[3*(i-1)+0][3*(j-1)+2] = copy.copy(hessian_x1z2)
                
                tmp_hess[3*(i-1)+1][3*(i-1)+0] = copy.copy(hessian_y1x1)
                tmp_hess[3*(i-1)+1][3*(i-1)+1] = copy.copy(hessian_y1y1)
                tmp_hess[3*(i-1)+1][3*(i-1)+2] = copy.copy(hessian_y1z1)
                tmp_hess[3*(i-1)+1][3*(j-1)+0] = copy.copy(hessian_y1x2)
                tmp_hess[3*(i-1)+1][3*(j-1)+1] = copy.copy(hessian_y1y2)
                tmp_hess[3*(i-1)+1][3*(j-1)+2] = copy.copy(hessian_y1z2)
                
                tmp_hess[3*(i-1)+2][3*(i-1)+0] = copy.copy(hessian_z1x1)
                tmp_hess[3*(i-1)+2][3*(i-1)+1] = copy.copy(hessian_z1y1)
                tmp_hess[3*(i-1)+2][3*(i-1)+2] = copy.copy(hessian_z1z1)
                tmp_hess[3*(i-1)+2][3*(j-1)+0] = copy.copy(hessian_z1x2)
                tmp_hess[3*(i-1)+2][3*(j-1)+1] = copy.copy(hessian_z1y2)
                tmp_hess[3*(i-1)+2][3*(j-1)+2] = copy.copy(hessian_z1z2)
                
                tmp_hess[3*(j-1)+0][3*(i-1)+0] = copy.copy(hessian_x2x1)
                tmp_hess[3*(j-1)+0][3*(i-1)+1] = copy.copy(hessian_x2y1)
                tmp_hess[3*(j-1)+0][3*(i-1)+2] = copy.copy(hessian_x2z1)
                tmp_hess[3*(j-1)+0][3*(j-1)+0] = copy.copy(hessian_x2x2)
                tmp_hess[3*(j-1)+0][3*(j-1)+1] = copy.copy(hessian_x2y2)
                tmp_hess[3*(j-1)+0][3*(j-1)+2] = copy.copy(hessian_x2z2)
                
                tmp_hess[3*(j-1)+1][3*(i-1)+0] = copy.copy(hessian_y2x1)
                tmp_hess[3*(j-1)+1][3*(i-1)+1] = copy.copy(hessian_y2y1)
                tmp_hess[3*(j-1)+1][3*(i-1)+2] = copy.copy(hessian_y2z1)
                tmp_hess[3*(j-1)+1][3*(j-1)+0] = copy.copy(hessian_y2x2)
                tmp_hess[3*(j-1)+1][3*(j-1)+1] = copy.copy(hessian_y2y2)
                tmp_hess[3*(j-1)+1][3*(j-1)+2] = copy.copy(hessian_y2z2)
                
                tmp_hess[3*(j-1)+2][3*(i-1)+0] = copy.copy(hessian_z2x1)
                tmp_hess[3*(j-1)+2][3*(i-1)+1] = copy.copy(hessian_z2y1)
                tmp_hess[3*(j-1)+2][3*(i-1)+2] = copy.copy(hessian_z2z1)
                tmp_hess[3*(j-1)+2][3*(j-1)+0] = copy.copy(hessian_z2x2)
                tmp_hess[3*(j-1)+2][3*(j-1)+1] = copy.copy(hessian_z2y2)
                tmp_hess[3*(j-1)+2][3*(j-1)+2] = copy.copy(hessian_z2z2)
            
                hessian = hessian + tmp_hess
            
            return hessian
        

        
            
        #--------------------------------------------------
        AFIR_e = e
        BPA_grad_list = g*0.0
        BPA_hessian = np.zeros((3*len(g), 3*len(g)))
        #debug_delta_BPA_grad_list = g*0.0
        
        
        for i in range(len(force_data["repulsive_potential_v2_well_scale"])):
            if force_data["repulsive_potential_v2_well_scale"][i] != 0.0:
                AFIR_e += calc_LJ_Repulsive_pot_v2(geom_num_list, force_data["repulsive_potential_v2_well_scale"][i], force_data["repulsive_potential_v2_dist_scale"][i], force_data["repulsive_potential_v2_length"][i], force_data["repulsive_potential_v2_const_rep"][i], force_data["repulsive_potential_v2_const_attr"][i], force_data["repulsive_potential_v2_order_rep"][i], force_data["repulsive_potential_v2_order_attr"][i], force_data["repulsive_potential_v2_center"][i], force_data["repulsive_potential_v2_target"][i], element_list)
                
                grad = calc_LJ_Repulsive_pot_v2_grad(geom_num_list, force_data["repulsive_potential_v2_well_scale"][i], force_data["repulsive_potential_v2_dist_scale"][i], force_data["repulsive_potential_v2_length"][i], force_data["repulsive_potential_v2_const_rep"][i], force_data["repulsive_potential_v2_const_attr"][i], force_data["repulsive_potential_v2_order_rep"][i], force_data["repulsive_potential_v2_order_attr"][i], force_data["repulsive_potential_v2_center"][i], force_data["repulsive_potential_v2_target"][i], element_list)
                BPA_grad_list += grad
                if self.FC_COUNT == -1:
                    pass
                elif iter % self.FC_COUNT == 0:
                    BPA_hessian = calc_LJ_Repulsive_pot_v2_hess(geom_num_list, force_data["repulsive_potential_v2_well_scale"][i], force_data["repulsive_potential_v2_dist_scale"][i], force_data["repulsive_potential_v2_length"][i], force_data["repulsive_potential_v2_const_rep"][i], force_data["repulsive_potential_v2_const_attr"][i], force_data["repulsive_potential_v2_order_rep"][i], force_data["repulsive_potential_v2_order_attr"][i], force_data["repulsive_potential_v2_center"][i], force_data["repulsive_potential_v2_target"][i], element_list, BPA_hessian)
            else:
                pass
        
        
        
        if force_data["gaussian_pot_energy"] != 0.0:
            AFIR_e += calc_gaussian_pot(geom_num_list, force_data["gaussian_pot_energy"], initial_geom_num_list)
            BPA_grad_list += calc_gaussian_pot_grad(geom_num_list, force_data["gaussian_pot_energy"], initial_geom_num_list)
        else:
            pass
        
        
        for i in range(len(force_data["repulsive_potential_dist_scale"])):
            if force_data["repulsive_potential_well_scale"][i] != 0.0:
                AFIR_e += calc_LJ_Repulsive_pot(geom_num_list, force_data["repulsive_potential_well_scale"][i], force_data["repulsive_potential_dist_scale"][i],  force_data["repulsive_potential_Fragm_1"][i], force_data["repulsive_potential_Fragm_2"][i], element_list)
                
                grad = calc_LJ_Repulsive_pot_grad(geom_num_list, force_data["repulsive_potential_well_scale"][i], force_data["repulsive_potential_dist_scale"][i],  force_data["repulsive_potential_Fragm_1"][i], force_data["repulsive_potential_Fragm_2"][i], element_list)
                BPA_grad_list += grad
                if self.FC_COUNT == -1:
                    pass
                elif iter % self.FC_COUNT == 0:
                    BPA_hessian = calc_LJ_Repulsive_pot_hess(geom_num_list, force_data["repulsive_potential_well_scale"][i], force_data["repulsive_potential_dist_scale"][i],  force_data["repulsive_potential_Fragm_1"][i], force_data["repulsive_potential_Fragm_2"][i], element_list, BPA_hessian)
            else:
                pass
        
        for i in range(len(force_data["keep_pot_spring_const"])):
            if force_data["keep_pot_spring_const"][i] != 0.0:
                AFIR_e += calc_keep_potential(geom_num_list[force_data["keep_pot_atom_pairs"][i][0]-1], geom_num_list[force_data["keep_pot_atom_pairs"][i][1]-1], force_data["keep_pot_spring_const"][i], force_data["keep_pot_distance"][i])
                
                grad_1, grad_2 = calc_keep_potential_grad(geom_num_list[force_data["keep_pot_atom_pairs"][i][0]-1], geom_num_list[force_data["keep_pot_atom_pairs"][i][1]-1], force_data["keep_pot_spring_const"][i], force_data["keep_pot_distance"][i])
                
                BPA_grad_list[force_data["keep_pot_atom_pairs"][i][0]-1] += np.array(grad_1, dtype="float64")
                BPA_grad_list[force_data["keep_pot_atom_pairs"][i][1]-1] += np.array(grad_2, dtype="float64")
                if self.FC_COUNT == -1:
                    pass
                elif iter % self.FC_COUNT == 0:
                    BPA_hessian = calc_keep_potential_hess(geom_num_list[force_data["keep_pot_atom_pairs"][i][0]-1], geom_num_list[force_data["keep_pot_atom_pairs"][i][1]-1], force_data["keep_pot_spring_const"][i], force_data["keep_pot_distance"][i], force_data["keep_pot_atom_pairs"][i][0], force_data["keep_pot_atom_pairs"][i][1], BPA_hessian)
            else:
                pass
                
        for i in range(len(force_data["anharmonic_keep_pot_spring_const"])):
            if force_data["anharmonic_keep_pot_potential_well_depth"][i] != 0.0:
                AFIR_e += calc_anharmonic_keep_potential(geom_num_list[force_data["anharmonic_keep_pot_atom_pairs"][i][0]-1], geom_num_list[force_data["anharmonic_keep_pot_atom_pairs"][i][1]-1], force_data["anharmonic_keep_pot_spring_const"][i], force_data["anharmonic_keep_pot_distance"][i], force_data["anharmonic_keep_pot_potential_well_depth"][i])
                
                
                grad_1, grad_2 = calc_anharmonic_keep_potential_grad(geom_num_list[force_data["anharmonic_keep_pot_atom_pairs"][i][0]-1], geom_num_list[force_data["anharmonic_keep_pot_atom_pairs"][i][1]-1], force_data["anharmonic_keep_pot_spring_const"][i], force_data["anharmonic_keep_pot_distance"][i], force_data["anharmonic_keep_pot_potential_well_depth"][i])
                
                BPA_grad_list[force_data["anharmonic_keep_pot_atom_pairs"][i][0]-1] += grad_1
                BPA_grad_list[force_data["anharmonic_keep_pot_atom_pairs"][i][1]-1] += grad_2
                
                if self.FC_COUNT != -1:
                    pass
                elif iter % self.FC_COUNT == 0:
                    BPA_hessian = calc_anharmonic_keep_potential_hess(geom_num_list[force_data["anharmonic_keep_pot_atom_pairs"][i][0]-1], geom_num_list[force_data["anharmonic_keep_pot_atom_pairs"][i][1]-1], force_data["anharmonic_keep_pot_spring_const"][i], force_data["anharmonic_keep_pot_distance"][i], force_data["anharmonic_keep_pot_potential_well_depth"][i], force_data["anharmonic_keep_pot_atom_pairs"][i][0], force_data["anharmonic_keep_pot_atom_pairs"][i][1], BPA_hessian)
            else:
                pass
            
            
        if len(geom_num_list) > 2:
            for i in range(len(force_data["keep_angle_spring_const"])):
                if force_data["keep_angle_spring_const"][i] != 0.0:
                    AFIR_e += calc_keep_angle(geom_num_list[force_data["keep_angle_atom_pairs"][i][0]-1], geom_num_list[force_data["keep_angle_atom_pairs"][i][1]-1], geom_num_list[force_data["keep_angle_atom_pairs"][i][2]-1], force_data["keep_angle_spring_const"][i], force_data["keep_angle_angle"][i])
                    
                    
                    grad_1, grad_2, grad_3 = calc_keep_angle_grad(geom_num_list[force_data["keep_angle_atom_pairs"][i][0]-1], geom_num_list[force_data["keep_angle_atom_pairs"][i][1]-1], geom_num_list[force_data["keep_angle_atom_pairs"][i][2]-1], force_data["keep_angle_spring_const"][i], force_data["keep_angle_angle"][i])
                    BPA_grad_list[force_data["keep_angle_atom_pairs"][i][0]-1] += grad_1
                    BPA_grad_list[force_data["keep_angle_atom_pairs"][i][1]-1] += grad_2
                    BPA_grad_list[force_data["keep_angle_atom_pairs"][i][2]-1] += grad_3
                    if self.FC_COUNT == -1:
                        pass
                    elif iter % self.FC_COUNT == 0:
                        pass 
                else:
                    pass
        else:
            pass
        
        
        if len(geom_num_list) > 3:
            for i in range(len(force_data["keep_dihedral_angle_spring_const"])):
                if force_data["keep_dihedral_angle_spring_const"][i] != 0.0:
                    AFIR_e += calc_keep_dihedral_angle(geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][0]-1], geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][1]-1], geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][2]-1], geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][3]-1], force_data["keep_dihedral_angle_spring_const"][i], force_data["keep_dihedral_angle_angle"][i])
                    
                    grad_1, grad_2, grad_3, grad_4 = calc_keep_dihedral_angle_grad(geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][0]-1], geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][1]-1], geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][2]-1], geom_num_list[force_data["keep_dihedral_angle_atom_pairs"][i][3]-1], force_data["keep_dihedral_angle_spring_const"][i], force_data["keep_dihedral_angle_angle"][i])
                    
                    BPA_grad_list[force_data["keep_dihedral_angle_atom_pairs"][i][0]-1] += grad_1
                    BPA_grad_list[force_data["keep_dihedral_angle_atom_pairs"][i][1]-1] += grad_2
                    BPA_grad_list[force_data["keep_dihedral_angle_atom_pairs"][i][2]-1] += grad_3
                    BPA_grad_list[force_data["keep_dihedral_angle_atom_pairs"][i][3]-1] += grad_4
                    if self.FC_COUNT == -1:
                        pass
                    elif iter % self.FC_COUNT == 0:
                        pass
                else:
                    pass
        else:
            pass

        
        for i in range(len(force_data["void_point_pot_spring_const"])):
            if force_data["void_point_pot_spring_const"][i] != 0.0:
                for j in force_data["void_point_pot_atoms"][i]:
                    AFIR_e += calc_void_point_pot(geom_num_list[j-1], np.array(force_data["void_point_pot_coord"][i], dtype="float64"), force_data["void_point_pot_spring_const"][i], force_data["void_point_pot_distance"][i], force_data["void_point_pot_order"][i])
                    
                    grad = calc_void_point_pot_grad(geom_num_list[j-1], np.array(force_data["void_point_pot_coord"][i], dtype="float64"), force_data["void_point_pot_spring_const"][i], force_data["void_point_pot_distance"][i], force_data["void_point_pot_order"][i])
                    BPA_grad_list[j-1] += grad
                    if self.FC_COUNT == -1:
                        pass
                    elif iter % self.FC_COUNT == 0:
                        BPA_hessian = calc_void_point_pot_hess(geom_num_list[j-1], np.array(force_data["void_point_pot_coord"][i], dtype="float64"), force_data["void_point_pot_spring_const"][i], force_data["void_point_pot_distance"][i], force_data["void_point_pot_order"][i], j, BPA_hessian)
            else:
                pass
        
        for i in range(len(force_data["AFIR_gamma"])):
            if force_data["AFIR_gamma"][i] != 0.0:
                AFIR_e += calc_AFIR_pot(geom_num_list, force_data["AFIR_gamma"][i],  force_data["AFIR_Fragm_1"][i], force_data["AFIR_Fragm_2"][i], element_list)
                
                BPA_grad_list += calc_AFIR_grad(geom_num_list, force_data["AFIR_gamma"][i],  force_data["AFIR_Fragm_1"][i], force_data["AFIR_Fragm_2"][i], element_list)
                if self.FC_COUNT == -1:
                    pass
                elif iter % self.FC_COUNT == 0:
                    BPA_hessian = calc_AFIR_hess(geom_num_list, force_data["AFIR_gamma"][i],  force_data["AFIR_Fragm_1"][i], force_data["AFIR_Fragm_2"][i], element_list, BPA_hessian)
            else:
                pass
                
        new_g = g + BPA_grad_list

        
        self.Model_hess.model_hess += BPA_hessian 
        
        #new_geometry:ang. 
        #AFIR_e:hartree
        return BPA_grad_list, AFIR_e, new_g
        
    def force_data_parser(self, args):
        def num_parse(numbers):
            sub_list = []
            
            sub_tmp_list = numbers.split(",")
            for sub in sub_tmp_list:                        
                if "-" in sub:
                    for j in range(int(sub.split("-")[0]),int(sub.split("-")[1])+1):
                        sub_list.append(j)
                else:
                    sub_list.append(int(sub))    
            return sub_list
        force_data = {}
       
        if len(args.repulsive_potential) % 4 != 0:
            print("invaild input (-rp)")
            sys.exit(0)
        
        force_data["repulsive_potential_well_scale"] = []
        force_data["repulsive_potential_dist_scale"] = []
        force_data["repulsive_potential_Fragm_1"] = []
        force_data["repulsive_potential_Fragm_2"] = []
        
        for i in range(int(len(args.repulsive_potential)/4)):
            force_data["repulsive_potential_well_scale"].append(float(args.repulsive_potential[4*i]))
            force_data["repulsive_potential_dist_scale"].append(float(args.repulsive_potential[4*i+1]))
            force_data["repulsive_potential_Fragm_1"].append(num_parse(args.repulsive_potential[4*i+2]))
            force_data["repulsive_potential_Fragm_2"].append(num_parse(args.repulsive_potential[4*i+3]))
        
        """
        parser.add_argument("-rpv2", "--repulsive_potential_v2", nargs="*",  type=str, default=['0.0','1.0','0.0','1','2','12','6' '1,2', '1-2'], help='Add LJ repulsive_potential based on UFF (ver.2) (ex.) [[well_scale] [dist_scale] [length] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] ...]')
        """
        if len(args.repulsive_potential_v2) % 9 != 0:
            print("invaild input (-rpv2)")
            sys.exit(0)
        
        force_data["repulsive_potential_v2_well_scale"] = []
        force_data["repulsive_potential_v2_dist_scale"] = []
        force_data["repulsive_potential_v2_length"] = []
        force_data["repulsive_potential_v2_const_rep"] = []
        force_data["repulsive_potential_v2_const_attr"] = []
        force_data["repulsive_potential_v2_order_rep"] = []
        force_data["repulsive_potential_v2_order_attr"] = []
        force_data["repulsive_potential_v2_center"] = []
        force_data["repulsive_potential_v2_target"] = []
        
        for i in range(int(len(args.repulsive_potential_v2)/9)):
            force_data["repulsive_potential_v2_well_scale"].append(float(args.repulsive_potential_v2[9*i+0]))
            force_data["repulsive_potential_v2_dist_scale"].append(float(args.repulsive_potential_v2[9*i+1]))
            force_data["repulsive_potential_v2_length"].append(float(args.repulsive_potential_v2[9*i+2]))
            force_data["repulsive_potential_v2_const_rep"].append(float(args.repulsive_potential_v2[9*i+3]))
            force_data["repulsive_potential_v2_const_attr"].append(float(args.repulsive_potential_v2[9*i+4]))
            force_data["repulsive_potential_v2_order_rep"].append(float(args.repulsive_potential_v2[9*i+5]))
            force_data["repulsive_potential_v2_order_attr"].append(float(args.repulsive_potential_v2[9*i+6]))
            force_data["repulsive_potential_v2_center"].append(num_parse(args.repulsive_potential_v2[9*i+7]))
            force_data["repulsive_potential_v2_target"].append(num_parse(args.repulsive_potential_v2[9*i+8]))

        
        if len(args.manual_AFIR) % 3 != 0:
            print("invaild input (-ma)")
            sys.exit(0)
        
        force_data["AFIR_gamma"] = []
        force_data["AFIR_Fragm_1"] = []
        force_data["AFIR_Fragm_2"] = []
        

        for i in range(int(len(args.manual_AFIR)/3)):
            force_data["AFIR_gamma"].append(float(args.manual_AFIR[3*i]))#kj/mol
            force_data["AFIR_Fragm_1"].append(num_parse(args.manual_AFIR[3*i+1]))
            force_data["AFIR_Fragm_2"].append(num_parse(args.manual_AFIR[3*i+2]))
        
        
        
        if len(args.anharmonic_keep_pot) % 4 != 0:
            print("invaild input (-akp)")
            sys.exit(0)
        
        force_data["anharmonic_keep_pot_potential_well_depth"] = []
        force_data["anharmonic_keep_pot_spring_const"] = []
        force_data["anharmonic_keep_pot_distance"] = []
        force_data["anharmonic_keep_pot_atom_pairs"] = []
        
        for i in range(int(len(args.anharmonic_keep_pot)/4)):
            force_data["anharmonic_keep_pot_potential_well_depth"].append(float(args.anharmonic_keep_pot[4*i]))#au
            force_data["anharmonic_keep_pot_spring_const"].append(float(args.anharmonic_keep_pot[4*i+1]))#au
            force_data["anharmonic_keep_pot_distance"].append(float(args.anharmonic_keep_pot[4*i+2]))#ang
            force_data["anharmonic_keep_pot_atom_pairs"].append(num_parse(args.anharmonic_keep_pot[4*i+3]))
        
        if len(args.keep_pot) % 3 != 0:
            print("invaild input (-kp)")
            sys.exit(0)
        
        force_data["keep_pot_spring_const"] = []
        force_data["keep_pot_distance"] = []
        force_data["keep_pot_atom_pairs"] = []
        
        for i in range(int(len(args.keep_pot)/3)):
            force_data["keep_pot_spring_const"].append(float(args.keep_pot[3*i]))#au
            force_data["keep_pot_distance"].append(float(args.keep_pot[3*i+1]))#ang
            force_data["keep_pot_atom_pairs"].append(num_parse(args.keep_pot[3*i+2]))
        
        if len(args.keep_angle) % 3 != 0:
            print("invaild input (-ka)")
            sys.exit(0)
        
        force_data["keep_angle_spring_const"] = []
        force_data["keep_angle_angle"] = []
        force_data["keep_angle_atom_pairs"] = []
        
        for i in range(int(len(args.keep_angle)/3)):
            force_data["keep_angle_spring_const"].append(float(args.keep_angle[3*i]))#au
            force_data["keep_angle_angle"].append(float(args.keep_angle[3*i+1]))#degrees
            force_data["keep_angle_atom_pairs"].append(num_parse(args.keep_angle[3*i+2]))
        
        if len(args.keep_dihedral_angle) % 3 != 0:
            print("invaild input (-kda)")
            sys.exit(0)
            
        force_data["keep_dihedral_angle_spring_const"] = []
        force_data["keep_dihedral_angle_angle"] = []
        force_data["keep_dihedral_angle_atom_pairs"] = []
        
        for i in range(int(len(args.keep_dihedral_angle)/3)):
            force_data["keep_dihedral_angle_spring_const"].append(float(args.keep_dihedral_angle[3*i]))#au
            force_data["keep_dihedral_angle_angle"].append(float(args.keep_dihedral_angle[3*i+1]))#degrees
            force_data["keep_dihedral_angle_atom_pairs"].append(num_parse(args.keep_dihedral_angle[3*i+2]))
        
        
        if len(args.void_point_pot) % 5 != 0:
            print("invaild input (-vpp)")
            sys.exit(0)
        
        force_data["void_point_pot_spring_const"] = []
        force_data["void_point_pot_distance"] = []
        force_data["void_point_pot_coord"] = []
        force_data["void_point_pot_atoms"] = []
        force_data["void_point_pot_order"] = []
        
        for i in range(int(len(args.void_point_pot)/5)):
            force_data["void_point_pot_spring_const"].append(float(args.void_point_pot[5*i]))#au
            force_data["void_point_pot_distance"].append(float(args.void_point_pot[5*i+1]))#ang
            coord = args.void_point_pot[5*i+2].split(",")
            force_data["void_point_pot_coord"].append(list(map(float, coord)))#ang
            force_data["void_point_pot_atoms"].append(num_parse(args.void_point_pot[5*i+3]))
            force_data["void_point_pot_order"].append(float(args.void_point_pot[5*i+4]))
        
        if len(args.gaussian_pot) > 1:
            print("invaild input (-gp)")
            sys.exit(0)
        
        for i in range(int(len(args.gaussian_pot))):
            force_data["gaussian_pot_energy"] = float(args.gaussian_pot[i])
        
        if len(args.fix_atoms) > 0:
            force_data["fix_atoms"] = num_parse(args.fix_atoms[0])
        else:
            force_data["fix_atoms"] = ""
        
        force_data["geom_info"] = num_parse(args.geom_info[0])
        
        force_data["opt_method"] = args.opt_method
        
        return force_data


    def main(self):
        force_data = self.force_data_parser(args)
        finish_frag = False
        geometry_list, element_list, electric_charge_and_multiplicity = self.make_geometry_list()
        file_directory = self.make_psi4_input_file(geometry_list, 0)
        #------------------------------------
        
        adam_m = []
        adam_v = []    
        for i in range(len(element_list)):
            adam_m.append(np.array([0,0,0], dtype="float64"))
            adam_v.append(np.array([0,0,0], dtype="float64"))        
        adam_m = np.array(adam_m, dtype="float64")
        adam_v = np.array(adam_v, dtype="float64")    
        self.Opt_params = Opt_calc_tmps(adam_m, adam_v, 0)
        self.Model_hess = Model_hess_tmp(np.eye(len(element_list*3)))
         

        #-----------------------------------
        with open(self.BPA_FOLDER_DIRECTORY+"input.txt", "w") as f:
            f.write(str(args))
        pre_AFIR_e = 0.0
        pre_e = 0.0
        pre_g = []
        for i in range(len(element_list)):
            pre_g.append(np.array([0,0,0], dtype="float64"))
       
        pre_move_vector = pre_g
        #-------------------------------------
        finish_frag = False
        exit_flag = False
        #-----------------------------------
        for iter in range(self.NSTEP):
            exit_file_detect = glob.glob(self.BPA_FOLDER_DIRECTORY+"*.txt")
            for file in exit_file_detect:
                if "end.txt" in file:
                    exit_flag = True
                    break
            if exit_flag:
                psi4.core.clean()
                break
            print("\n# ITR. "+str(iter)+"\n")  
            e, g, geom_num_list, finish_frag = self.psi4_calculation(file_directory, element_list, 
            electric_charge_and_multiplicity, iter)
            
            if iter == 0:
                initial_geom_num_list = geom_num_list
                pre_geom = initial_geom_num_list
                
            #--------------------geometry info
            if len(force_data["geom_info"]) > 1:
                CSI = CalculationStructInfo
                data_list, data_name_list = CSI.Data_extract(glob.glob(file_directory+"/*.xyz")[0], force_data["geom_info"])
                if iter == 0:
                    with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:
                        f.write(",".join(data_name_list)+"\n")
                
                with open(self.BPA_FOLDER_DIRECTORY+"geometry_info.csv","a") as f:    
                    f.write(",".join(list(map(str,data_list)))+"\n")
            #-------------------energy profile 
            if iter == 0:
                with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                    f.write("energy [hartree] \n")
            with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","a") as f:
                f.write(str(e)+"\n")
            #-------------------
            if finish_frag:#If QM calculation doesnt end, the process of this program is terminated. 
                break   
            
            _, AFIR_e, new_g = self.calc_biaspot(e, g, geom_num_list, element_list, force_data, pre_g, iter, initial_geom_num_list)#new_geometry:ang.
            
            #if iter == 0:
            #    Model_hess = Model_hess_tmp(np.eye(len(element_list*3))+np.dot((new_g.reshape(len(new_g)*3,1)), (new_g.reshape(1,len(new_g)*3))))
            #    print(Model_hess.model_hess)
            #----------------------------
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_g[j-1] = copy.deepcopy(new_g[j-1]*self.bohr2angstroms*0.0)
            #----------------------------
            
            CMV = CalculateMoveVector(self.DELTA, self.Opt_params, self.Model_hess, self.FC_COUNT, self.temperature)
            new_geometry, move_vector, iter, Opt_params, Model_hess = CMV.calc_move_vector(iter, geom_num_list, new_g, force_data["opt_method"], pre_g, pre_geom, AFIR_e, pre_AFIR_e, pre_move_vector, e, pre_e, initial_geom_num_list)
            self.Opt_params = Opt_params
            self.Model_hess = Model_hess

            self.ENERGY_LIST_FOR_PLOTTING.append(e*self.hartree2kcalmol)
            self.AFIR_ENERGY_LIST_FOR_PLOTTING.append(AFIR_e*self.hartree2kcalmol)
            self.NUM_LIST.append(int(iter))
            #----------------------------
            print("caluculation results (unit a.u.):")
            print("OPT method            : {} ".format(force_data["opt_method"]))
            print("                         Value                         Threshold ")
            print("ENERGY                : {:>15.12f} ".format(e))
            print("BIAS  ENERGY          : {:>15.12f} ".format(AFIR_e))
            print("Maxinum  Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(new_g.max()), self.MAX_FORCE_THRESHOLD))
            print("RMS      Force        : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(np.square(new_g).mean())), self.RMS_FORCE_THRESHOLD))
            print("Maxinum  Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(move_vector.max()), self.MAX_DISPLACEMENT_THRESHOLD))
            print("RMS      Displacement : {0:>15.12f}             {1:>15.12f} ".format(abs(np.sqrt(np.square(move_vector).mean())), self.RMS_DISPLACEMENT_THRESHOLD))
            print("ENERGY SHIFT          : {:>15.12f} ".format(e - pre_e))
            print("BIAS ENERGY SHIFT     : {:>15.12f} ".format(AFIR_e - pre_AFIR_e))
            
            
            if abs(new_g.max()) < self.MAX_FORCE_THRESHOLD and abs(np.sqrt(np.square(new_g).mean())) < self.RMS_FORCE_THRESHOLD and  abs(move_vector.max()) < self.MAX_DISPLACEMENT_THRESHOLD and abs(np.sqrt(np.square(move_vector).mean())) < self.RMS_DISPLACEMENT_THRESHOLD:#convergent criteria
                break
            #-------------------------
            print("\ngeometry:")
            if len(force_data["fix_atoms"]) > 0:
                for j in force_data["fix_atoms"]:
                    new_geometry[j-1] = copy.deepcopy(initial_geom_num_list[j-1]*self.bohr2angstroms)
            #----------------------------
            
            pre_AFIR_e = AFIR_e#Hartree
            pre_e = e
            pre_g = new_g#Hartree/Bohr
            pre_geom = geom_num_list#Bohr
            pre_move_vector = move_vector
            
            geometry_list = self.make_geometry_list_2(new_geometry, element_list, electric_charge_and_multiplicity)
            file_directory = self.make_psi4_input_file(geometry_list, iter+1)
            
        self.sinple_plot(self.NUM_LIST, self.ENERGY_LIST_FOR_PLOTTING, self.AFIR_ENERGY_LIST_FOR_PLOTTING, file_directory)
        self.xyz_file_make()
        #-----------------------
        
        local_max_energy_list_index = argrelextrema(np.array(self.ENERGY_LIST_FOR_PLOTTING), np.greater)
        with open(self.BPA_FOLDER_DIRECTORY+"approx_TS.txt","w") as f:
            for j in local_max_energy_list_index[0].tolist():
                f.write(str(self.NUM_LIST[j])+"\n")
        
        inverse_energy_list = (-1)*np.array(self.ENERGY_LIST_FOR_PLOTTING, dtype="float64")
        local_min_energy_list_index = argrelextrema(np.array(inverse_energy_list), np.greater)
        with open(self.BPA_FOLDER_DIRECTORY+"approx_EQ.txt","w") as f:
            for j in local_min_energy_list_index[0].tolist():
                f.write(str(self.NUM_LIST[j])+"\n")
        
        with open(self.BPA_FOLDER_DIRECTORY+"energy_profile.csv","w") as f:
            f.write("ITER.,energy[kcal/mol]\n")
            for i in range(len(self.ENERGY_LIST_FOR_PLOTTING)):
                f.write(str(i)+","+str(self.ENERGY_LIST_FOR_PLOTTING[i] - self.ENERGY_LIST_FOR_PLOTTING[0])+"\n")
        
       
        #----------------------
        print("Complete...")
        return
        


class Opt_calc_tmps:
    def __init__(self, adam_m, adam_v, adam_count, eve_d_tilde=0.0):
        self.adam_m = adam_m
        self.adam_v = adam_v
        self.adam_count = 1 + adam_count
        self.eve_d_tilde = eve_d_tilde
            
class Model_hess_tmp:
    def __init__(self, model_hess, momentum_disp=0, momentum_grad=0):
        self.model_hess = model_hess
        self.momentum_disp = momentum_disp
        self.momentum_grad = momentum_grad

class CalculationStructInfo:
    def __init__():
        return
    
    def calculate_distance(self, atom1, atom2):
        atom1, atom2 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64")
        distance = np.linalg.norm(atom2 - atom1)
        return distance


    def calculate_bond_angle(self, atom1, atom2, atom3):
        atom1, atom2, atom3 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64"), np.array(atom3, dtype="float64")
        vector1 = atom1 - atom2
        vector2 = atom3 - atom2

        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cos_angle)
        angle_deg = np.degrees(angle)

        return angle_deg
        
    def calculate_dihedral_angle(self, atom1, atom2, atom3, atom4):
        atom1, atom2, atom3, atom4 = np.array(atom1, dtype="float64"), np.array(atom2, dtype="float64"), np.array(atom3, dtype="float64"), np.array(atom4, dtype="float64")
        
        a1 = atom2 - atom1
        a2 = atom3 - atom2
        a3 = atom4 - atom3

        v1 = np.cross(a1, a2)
        v1 = v1 / np.linalg.norm(v1, ord=2)
        v2 = np.cross(a2, a3)
        v2 = v2 / np.linalg.norm(v2, ord=2)
        porm = np.sign((v1 * a3).sum(-1))
        angle = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)
        if not porm == 0:
            angle = angle * porm
            
        dihedral_angle_deg = np.degrees(angle)

        return dihedral_angle_deg
        

    def read_xyz_file(self, file_name):
        with open(file_name,"r") as f:
            words = f.readlines()
        mole_struct_list = []
            
        for word in words[1:]:
            mole_struct_list.append(word.split())
        return mole_struct_list

    def Data_extract(self, file, atom_numbers):
        data_list = []
        data_name_list = [] 
         
        
        
        mole_struct_list = self.read_xyz_file(file)
        DBD_list = []
        DBD_name_list = []
        print(file)
        if len(atom_numbers) > 1:
            for a1, a2 in list(itertools.combinations(atom_numbers,2)):
                try:
                    distance = self.calculate_distance(mole_struct_list[int(a1) - 1][1:4], mole_struct_list[int(a2) - 1][1:4])
                    DBD_name_list.append("Distance ("+str(a1)+"-"+str(a2)+")  [ang.]")
                    DBD_list.append(distance)
                        
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Distance ("+str(a1)+"-"+str(a2)+")  [ang.]")
                    DBD_list.append("nan")
                
        if len(atom_numbers) > 2:
            for a1, a2, a3 in list(itertools.permutations(atom_numbers,3)):
                try:
                    bond_angle = self.calculate_bond_angle(mole_struct_list[int(a1)-1][1:4], mole_struct_list[int(a2)-1][1:4], mole_struct_list[int(a3)-1][1:4])
                    DBD_name_list.append("Bond_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+") [deg.]")
                    DBD_list.append(bond_angle)
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Bond_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+") [deg.]")
                    DBD_list.append("nan")            
        
        if len(atom_numbers) > 3:
            for a1, a2, a3, a4 in list(itertools.permutations(atom_numbers,4)):
                try:
                    dihedral_angle = self.calculate_dihedral_angle(mole_struct_list[int(a1)-1][1:4], mole_struct_list[int(a2)-1][1:4],mole_struct_list[int(a3)-1][1:4], mole_struct_list[int(a4)-1][1:4])
                    DBD_name_list.append("Dihedral_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+"-"+str(a4)+") [deg.]")
                    DBD_list.append(dihedral_angle)
                except Exception as e:
                    print(e)
                    DBD_name_list.append("Dihedral_angle ("+str(a1)+"-"+str(a2)+"-"+str(a3)+"-"+str(a4)+") [deg.]")
                    DBD_list.append("nan")        

        data_list = DBD_list 
        
        data_name_list = DBD_name_list    
        return data_list, data_name_list


if __name__ == "__main__":
    args = parser()
    bpa = BiasPotentialAddtion(args)
    bpa.main()

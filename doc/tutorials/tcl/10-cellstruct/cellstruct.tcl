# Copyright (C) 2010,2011,2012,2013,2014,2015,2016 The ESPResSo project
#  
# This file is part of ESPResSo.
#  
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#  
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#  
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#
#############################################################
#                                                           #
#  Sample System  Lennard Jones Liquid                      #
#                                                           #
#    LJ System Studied in the Book                          # 
#    Understanding Molecular Simulations                    # 
#    Frankel & Smith , 2nd Edition                          # 
#    Chapter 4, Case Study 4                                # 
#                                                           # 
#############################################################
# Source (call) functions to be used
#source lj_functions.tcl


puts " "
puts "======================================================="
puts "=                        cellstruct.tcl       ="
puts "======================================================="
puts " "
puts "Espresso Code Base : \n[code_info]\n"
puts " "
puts " "


cellsystem domain_decomposition  -no_verlet_list
#############################################################
#  Parameters                                               #
#############################################################
# System parameters
#############################################################
#setmd periodic 1 0 0

# 108 particles
set n_part  108

# Integration parameters
#############################################################
set skin 0.3
setmd skin  $skin

# Other parameters
#############################################################
#setting a seed for the random number generator
#expr srand([pid])

#############################################################
#  Setup System                                             #
#############################################################


# Particle setup
#############################################################

set density 0.2
set box_length [expr pow($n_part/$density,1.0/3.0)+2*$skin]
puts " density = $density box_length = $box_length"
setmd box $box_length $box_length $box_length
#for {set i 0} {$i < $n_part} {incr i} {
#   set pos_x [expr rand()*$box_length]
#   set pos_y [expr rand()*$box_length]
#   set pos_z [expr rand()*$box_length]
#   part $i pos $pos_x $pos_y $pos_z q 0.0 type 0
#}
######################################################################################
#if { [file exists data/config.pdb]==0 } {
#    exec mkdir data
#    exec touch data/config.pdb
#    puts "Creating directory data and writing simulation data to data/config.pdb" 
#} else {
#    puts "Writing simlulation data to data/config.pdb" 
#}
######################################################################################
#writepdb data/config.pdb
#writevtk data/part.vtk

# Interaction parameters
#############################################################

set lj1_eps     1.0
set lj1_sig     1.0
set lj1_cut     2.5
# set lj1_shift   [expr -4.0*$lj1_eps*(pow(1.0/$lj1_cut,12)-pow(1.0/$lj1_cut,6))]
set lj1_shift   [expr -(pow(1.0/$lj1_cut,12)-pow(1.0/$lj1_cut,6))]
set lj1_offset  0.0
inter 0 0 lennard-jones $lj1_eps $lj1_sig $lj1_cut $lj1_shift $lj1_offset

exit

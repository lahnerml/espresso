# initialize domain decomposition; do not use verlet lists

# setup particles
set n_part 5

setmd skin 0.1

# define interaction
inter 0 0 lennard-jones 1.0 0.1 0.8 auto

# variables
set radius 1.0; # Radius of cylinders

set box_x 10
set box_y 40
set box_z 4
setmd box_l $box_x $box_y $box_z
setmd periodic 1 1 1

cellsystem domain_decomposition -no_verlet_list

set v_part -0.0381

expr srand(0)
part 0 pos [expr 0.5 * ($box_x - 0.5) + rand()] [expr $box_y - 1 - 4.0 * rand()] [expr 0.5 * $box_z] q 0.0 type 0 v 0 $v_part 0
for {set i 1} {$i < $n_part} {incr i} {
  set pos_x [expr 0.5 * ($box_x - 0.5) + rand()]
  set pos_y [expr $box_y - 1 - 4.0 * rand()]
  set pos_z [expr 0.5 * $box_z]
  #part $i pos $pos_x $pos_y $pos_z q 0.0 type 0 ext_force 0.1 0 0
  part $i pos $pos_x $pos_y $pos_z q 0.0 type 0 v 0 $v_part 0
}

set dt 0.001
setmd time_step $dt

# read refinement levels from command line or use default values
if { $argc == 3 } {
    set min_level [lindex $argv 0]
    set max_level [lindex $argv 1]
    set steps [lindex $argv 2]
} elseif { $argc == 2 } {
    set min_level [lindex $argv 0]
    set max_level [lindex $argv 1]
    set steps [expr pow(2, ($max_level - 4)) * 1024]
} else {
    set min_level 0
    set max_level 2
    set steps 256
}
set agrid_min [expr 1./pow(2, $min_level)]
set agrid_max [expr 1./pow(2, $max_level)]
puts [concat "Using level ${min_level} (h = ${agrid_min}), refining up to" \
             " level ${max_level} (h = ${agrid_max})"]

# set output directories
set username $::env(USER)
if {"neon.informatik.uni-stuttgart.de" == [info hostname] } {
  set folder /data/scratch-ssd1/${username}/05_fibers/level_${max_level}/
} elseif {[string match "lapsgs*" [info hostname]]} {
  set folder ""
} elseif {[string match "pcsgs*" [info hostname]]} {
  set folder ""
} elseif {[string match "kepler*" [info hostname]]} {
  set folder ""
} elseif {[string match "*.informatik.uni-stuttgart.de" [info hostname]]} {
  set folder /data/scratch/${username}/05_fibers/level_${max_level}/
} else {
  set folder ""
}
set filename ${folder}05_fibers_min_${min_level}_max_${max_level}

thermostat off

puts "Mindist: [analyze mindist]"
minimize_energy 10.0 100 0.1 0.01
kill_particle_motion
kill_particle_forces
puts "Mindist: [analyze mindist]"

# init p4est
lbadapt-init $min_level
lbadapt_set_max_level $max_level

#lbfluid cpu agrid $agrid_max dens 0.0000068 visc 0.152 tau $dt friction 0.000007
lbfluid cpu agrid $agrid_max dens 1. visc 0.032 tau $dt friction 0.5

lbboundary wall dist [expr -1 * ($box_y-$agrid_min)] normal 0.0 -1.0 0.0 velocity 0 $v_part 0
lbboundary wall dist [expr $agrid_min] normal 0.0 1.0 0.0 velocity 0 $v_part 0

lbboundary cylinder center 1 23 2 axis 0 0 1 radius $radius length [expr $box_z *0.5] direction 1 type 9 velocity 0 0 0
lbboundary cylinder center 8 27 2 axis 0 0 1 radius $radius length [expr $box_z *0.5] direction 1 type 9 velocity 0 0 0
lbboundary cylinder center 4 15 2 axis 0 0 1 radius $radius length [expr $box_z *0.5] direction 1 type 9 velocity 0 0 0
lbboundary cylinder center 5.5 24 2 axis 0 0 1 radius $radius length [expr $box_z *0.5] direction 1 type 9 velocity 0 0 0
lbboundary cylinder center 9.5 19 2 axis 0 0 1 radius $radius length [expr $box_z *0.5] direction 1 type 9 velocity 0 0 0

lbadapt-exclude-bnd-from-geom-ref 0
lbadapt-exclude-bnd-from-geom-ref 1

# perform a geometric refinement: recursivly refine all quadrants whose
# midpoint is closer to an obstacle than 0.6 * sqrt(3) * $cell_sidelength
lbadapt-geom-ref

# print boundaries and refinement pattern
lbfluid print vtk boundary ${filename}_boundary.vtk

lbadapt-reset-fluid

set itermax [expr $steps / pow(2, $max_level - $min_level)]
for {set i 0} {$i < $itermax} {incr i} {
  puts "Performing integration step $i of $itermax"
  md_vtk ${filename}_md_part_${i}.vtk
  lbfluid print vtk velocity ${filename}_vel_${i}.vtk
  integrate [expr int(100 * pow(2, $max_level - $min_level))]
}

lbfluid print vtk velocity ${filename}_vel_${i}.vtk
md_vtk ${filename}_md_part_${i}.vtk

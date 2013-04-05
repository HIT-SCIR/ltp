package MAPPING;

#Copyright (C) 2004 Jesus Gimenez and Lluis Marquez

#This library is free software; you can redistribute it and/or
#modify it under the terms of the GNU Lesser General Public
#License as published by the Free Software Foundation; either
#version 2.1 of the License, or (at your option) any later version.

#This library is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#Lesser General Public License for more details.

#You should have received a copy of the GNU Lesser General Public
#License along with this library; if not, write to the Free Software
#Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

use strict;
use IO;
use Data::Dumper;
use SVMTool::COMMON;

# ==================================== MAPPING =========================================

sub get_size
{
   my $file = shift;

   my $FILE = new IO::File("< $file") or die "Couldn't open input file: $file\n";
   my $size;

   while (defined(my $line = $FILE->getline())) {
      $size++;
   }

   $FILE->close();

   return $size;
}

sub filter_mapping
{
   #description _ creates a MAPPING file
   #param1 _ SAMPLE MAPPING structure reference
   #param2 _ mapping min frequency
   #param3 _ mapping max number of entries
   #param4 _ report file
   #param5 _ verbosity [0..3]

   my $rM = shift;
   my $minfr = shift;
   my $mapsize = shift;
   my $report = shift;
   my $verbose = shift;

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "FILTERING MAPING (minfreq = $minfr :: maxmapsize = $mapsize)\n"); }

   if ($verbose > $COMMON::verbose2) { COMMON::report($report, "MAPPING SIZE (1) -> size == ".scalar(keys %{$rM})."\n"); }

   if ($minfr > 1) {
      $rM = reduce_mapping($rM, $minfr);
      if ($verbose > $COMMON::verbose2) { COMMON::report($report, "REDUCING MAPPING ($minfr) -> size == ".scalar(keys %{$rM})."\n"); }
   }
   else { $minfr = 1; }
   my $i = $minfr + 1;
   while (scalar(keys %{$rM}) > $mapsize) {
       $rM = reduce_mapping($rM, $i);
       if ($verbose > $COMMON::verbose2) { COMMON::report($report, "REDUCING MAPPING ($i) -> size == ".scalar(keys %{$rM})."\n"); }
       $i++;
   }

   return $rM;
}

sub reduce_mapping
{
   #description _ creates a MAPPING file
   #param1 _ mapping hash reference
   #param2 _ filtering coefficient

   my $rM = shift;
   my $f = shift;

   my %M;

   foreach my $k (keys %{$rM}) {
      if ($rM->{$k} >= $f) { $M{$k} = $rM->{$k}; }
   }

   %{$rM} = ();

   return(\%M);
}

sub make_mapping
{
   #description _ creates a MAPPING file
   #param1  _ SAMPLE SET filename
   #param2 _ report file
   #param3 _ verbosity [0..3]
   #@return _ SAMPLE MAPPING structure reference

   my $smplset = shift;
   my $report = shift;
   my $verbose = shift;

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "MAKING MAPPING from <$smplset>..."); }


   my $SMPLF = new IO::File("< $smplset") or die "Couldn't open input file: $smplset\n";
   
   my %MAP;
   while (defined(my $line = $SMPLF->getline())) {
      chomp($line);
      my @entry = split(" ", $line);
      my $size = scalar(@entry);
      my $i = 1;
      while ($i < $size) {
	 if (exists($MAP{$entry[$i]})) { $MAP{$entry[$i]}++; }
	 else { $MAP{$entry[$i]} = 1; }
	 $i++;
      }            
   }

   $SMPLF->close();

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, " [DONE]\n"); }

   return \%MAP;
}

sub write_mapping
{
   #description _ creates a MAPPING file
   #param1 _ SAMPLE MAPPING structure reference
   #param2 _ SAMPLE MAPPING filename
   #param3 _ report file
   #param4 _ verbosity [0..3]

   my $rM = shift;
   my $map = shift;
   my $report = shift;
   my $verbose = shift;

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "WRITING MAPPING <$map>..."); }

   my $MAPF = new IO::File("> $map") or die "Couldn't open output file: $map\n";

   my $i = 1;
   my $freq = -1;
   foreach my $k (sort keys %{$rM}) {
      if (($freq == -1) or ($rM->{$k} < $freq)) { $freq = $rM->{$k}; }
      print $MAPF $i." ".$k." ".$rM->{$k}."\n";
      $i++;
   }

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, " [FREQ = ".$freq."] :: [SIZE = ".(scalar(keys %{$rM}))."] [DONE]\n"); }

   $MAPF->close();
}

sub map_set
{
   #description _ MAPS a set given set file and a MAPPING file
   #param1 _ SAMPLE SET filename
   #param2 _ SAMPLE MAPPING filename
   #param3 _ TARGET SAMPLE MAPPED filename
   #param4 _ report file
   #param5 _ verbosity [0..3]

   my $smplset = shift;
   my $smplmap = shift;
   my $smpldsf = shift;
   my $report = shift;
   my $verbose = shift;

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "MAPPING DATA FEATURES from <$smplset> to <$smpldsf>..."); }
   if ($verbose > $COMMON::verbose2) { COMMON::report($report, "\n"); }


   my $SMPLF = new IO::File("< $smplset") or die "Couldn't open input file: $smplset\n";
   my $MAPF = new IO::File("< $smplmap") or die "Couldn't open input file: $smplmap\n";
   my $DSFF = new IO::File("> $smpldsf") or die "Couldn't open output file: $smpldsf\n";
   
   my $MAP = read_mapping_B($smplmap);

   my $iter = 0;
   while (defined(my $line = $SMPLF->getline())) {
      chomp($line);
      my @entry = split(" ", $line);
      my $i = 1;
      my %features;
      while ($i < scalar(@entry)) { # JOIN + UNIQ
	 if (exists($MAP->{$entry[$i]})) { $features{$MAP->{$entry[$i]}} = 1; }
	 $i++;
      }
      my @sample;
      push(@sample, $entry[0]);
      foreach my $f (sort {$a <=> $b} keys %features) { # SORT
	 push(@sample, $f.":".$features{$f});
      }
      print $DSFF join(" ", @sample), "\n";
      $iter++;
      if ($verbose > $COMMON::verbose1) { COMMON::show_progress($iter, $COMMON::progress1, $COMMON::progress2); }
   }

   if ($verbose > $COMMON::verbose1) { COMMON::report($report, "..$iter [DONE]\n"); }

   $SMPLF->close();
   $DSFF->close();
}

sub read_mapping
{
   #description _ reads a mapping file from disk onto memory
   #param1 _ MAPPING FILE                (input)
   #@return _ mapping hash reference (*) [see options...] 
   #          return only reverse mapping         <n>     -> attrib.name         

   my $smplmap = shift;

   my $SMPLMAP = new IO::File("< $smplmap") or die "Couldn't open input file: $smplmap\n";
   my %MAP;
   while (defined(my $m = $SMPLMAP->getline())) {
       chomp $m;
       my @line = split(" ",$m);
       $MAP{$line[0]} = $line[1];
   }
   $SMPLMAP->close();

   return \%MAP;
}

sub read_mapping_B
{
   #description _ reads a mapping file from disk onto memory
   #param1 _ MAPPING FILE                (input)
   #@return _ mapping hash reference (*) [see options...] 
   #          return mapping         attrib.name         ->    <n>

   my $smplmap = shift;

   my $SMPLMAP = new IO::File("< $smplmap") or die "Couldn't open input file: $smplmap\n";
   my %MAP;
   while (defined(my $m = $SMPLMAP->getline())) {
       chomp $m;
       my @line = split(" ",$m);
       $MAP{$line[1]} = $line[0];
   }
   $SMPLMAP->close();

   return \%MAP;
}

sub read_mapping_C
{
   #description _ reads a mapping file from disk onto memory
   #param1  _ MAPPING FILE                (input)
   #@return _ mapping hash reference (*) [see options...] 
   #          return only reverse mapping         <n>     -> attrib.name         

   my $smplmap = shift;

   my $SMPLMAP = new IO::File("< $smplmap") or die "Couldn't open input file: $smplmap\n";
   my %MAP;
   while (defined(my $m = $SMPLMAP->getline())) {
       chomp $m;
       my @line = split(" ",$m);
       $MAP{$line[1]} = $line[2];
   }
   $SMPLMAP->close();

   return \%MAP;
}

sub map_sample
{
   #description _ runs a given test/val set, given a mapping, through the
   #              given models (assuming they've already been learned).
   #param1 _ SAMPLE attribute list reference                     (input)
   #param2 _ SAMPLE MAPPING hash reference                       (input)
   #@return _ binarized sample                                   (output)
 
   my $rsample = shift;
   my $rmap = shift;

   my $pos = shift(@{$rsample});
   my @dsf;

   push(@dsf, $pos);
   foreach my $elem (@{$rsample}) {
      my $bin;
      if (exists($rmap->{$elem})) { $bin = $rmap->{$elem}.":1"; push(@dsf, $bin); }
   }

   return \@dsf;
}


sub load_mapping #OUT OF DATE
{
   #description _ loads a MAPPING file (discarding attributes under a certain frequency)
   #param1 _ SOURCE SAMPLE MAPPING filename
   #param2 _ min frequency

   my $smplmap = shift;
   my $min = shift;
 
   print "LOADING MAPPING <$smplmap>\n";

   my $MAP = new IO::File("< $smplmap") or die "Couldn't open input file: $smplmap\n";
   my @M;
   my $i = 1;
   while (defined(my $line = $MAP->getline())) {
      my @entry = split(" ", $line);
      my @l = ($entry[1], $entry[2]);
      if ( $entry[2] >= $min) { $M[$i - 1] = \@l; $i++; }
   }
   $MAP->close();

   print "-> size == ", scalar(@M), "\n";

   return(\@M);
}

sub save_mapping #OUT OF DATE
{
   #description _ saves a MAPPING file
   #param1 _ TARGET SAMPLE MAPPING filename
   #param2 _ mapping list reference

   my $smplmap = shift;
   my $rM = shift;

   print "SAVING MAPPING <$smplmap>\n";

   my $MAP = new IO::File("> $smplmap") or die "Couldn't open output file: $smplmap\n";

   my $i = 0;
   while ($i < scalar(@{$rM})) {
       print $MAP $i+1, " ", $rM->[$i]->[0], " ", $rM->[$i]->[1], "\n";
       $i++;
   }
   $MAP->close();

   return (scalar(@{$rM}));
}

sub extend
{
   #description _ extends a MAPPING file
   #param1 _ input mapping filename
   #param2 _ set of new features (hash ref)
   #param3 _ extended mapping filename

   my $map = shift;
   my $rNEW = shift;
   my $newmap = shift;

   my $rM = read_mapping_C($map);

   foreach my $k (keys %{$rNEW}) {
      if (exists($rM->{$k})) { die "FEATURE ALREADY EXISTED!!\n"; }
      else { $rM->{$k} = $rNEW->{$k}; }
   }

   write_mapping($rM, $newmap, "", 0);
}

1;

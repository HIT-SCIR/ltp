package ENTRY;

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
use Data::Dumper;
use SVMTool::COMMON;

sub read_sentence_stdin
{  #description _ reads input from STDIN
   #param1  _ EOS usage? (1: YES, 0: NO)
   #@return _ (stdin [STDIN sentence as is], in [STDIN sentence prepared for SVMTool], stop condition)

   my $EOS = shift;

   my @STDIN_LR;
   my @INPUT_LR;
   my $stop = 0;
   while (!$stop) {
      if (my $entry = <STDIN>) {
         chomp($entry);
         my @line = split($COMMON::in_valseparator, $entry);
         if ($EOS) {
            if ($entry ne $COMMON::SMARK) { push(@STDIN_LR, $entry); }
            if (($line[0] ne $COMMON::IGNORE) and ($line[0] ne "") and ($entry ne $COMMON::SMARK)) {
               my $input = new ENTRY($line[0], $COMMON::emptypos, 0, \@line);
               push(@INPUT_LR, $input);
            }
            $stop = ($entry eq $COMMON::SMARK);
	 }
         else {
            push(@STDIN_LR, $entry);
            if (($line[0] ne $COMMON::IGNORE) and ($line[0] ne "")) {
               my $input = new ENTRY($line[0], $COMMON::emptypos, 0, \@line);
               push(@INPUT_LR, $input);
            }
            $stop = (COMMON::end_of_sentence($line[0]));
	 }
      }
      else { $stop = 2; }
   }

   return (\@STDIN_LR, \@INPUT_LR, $stop == 2);
}

sub read_sentence_file
{  #description _ reads input from STDIN
   #param1 _ filehandle
   #@return _ (sentence as is, sentence prepared for SVMT, stop condition)

   my $INPUT = shift;

   my @STDIN_LR;
   my @INPUT_LR;
   my $stop = 0;
   while (!$stop) {
      if (defined(my $entry = $INPUT->getline())) {
         chomp($entry);
         my @line = split($COMMON::in_valseparator, $entry);
         push(@STDIN_LR, $entry);
         if (($line[0] ne $COMMON::IGNORE) and ($line[0] ne "")) {
            my $input = new ENTRY($line[0], $COMMON::emptypos, 0, \@line);
            push(@INPUT_LR, $input);
         }
         $stop = COMMON::end_of_sentence($line[0]);
      }
      else { $stop = 2; }
   }

   return (\@STDIN_LR, \@INPUT_LR, $stop == 2);
}

# ======================================================================================
# =============================== ENTRY OBJECT METHODS =================================
# ======================================================================================

sub new
{
    #description _ creates a new ENTRY object
    #param1 _ word
    #param2 _ pos
    #param3 _ pos-prediction hash reference
    #param4 _ whole input line (list)

    my $class = shift;     #implicit parameter
    my $word = shift;
    my $pos = shift;
    my $pp = shift;
    my $line = shift;

    my @l;
    if (defined(@{$line})) {
       my $j = 0;
       while ($j < scalar(@{$line})) {
          push(@l, $line->[$j]);
          $j++;
       }
    }
    my $rout = { word => $word, pos => $pos, pp => $pp, COLS => \@l };
    bless $rout, $class;

    return $rout;
}

sub print
{
    #description _ prints the contents of an ENTRY object

    my $rout = shift;  #implicit parameter

    print "\n====================================\n";
    print "WORD: \t", $rout->{word}, "\n";
    print "POS: \t", $rout->{pos}, "\n";
    if (defined($rout->{pp})) {
	print "PREDICTIONS:\n";
       foreach my $elem (keys %{$rout->{pp}}) {
	   print "\t", $elem, " -> ", $rout->{pp}->{$elem}, "\n";
       }
    }
}

sub sort_pos
{
    #description _ returns a sorted list of predictions

    my $rout = shift;  #implicit parameter
    my @sorted;
    my %REV;

    if (defined($rout->{pp})) {
       foreach my $elem (keys %{$rout->{pp}}) {
	   $REV{$rout->{pp}->{$elem}} = $elem;
       }
       foreach my $pred (sort{$b <=> $a} keys %REV) {
           push(@sorted, $REV{$pred});
       }
    }

    return \@sorted;
}

sub sort_pred
{
    #description _ returns a sorted list of predictions

    my $rout = shift;  #implicit parameter
    my @sorted;
    my %REV;

    if (defined($rout->{pp})) {
       foreach my $elem (keys %{$rout->{pp}}) {
	   $REV{$rout->{pp}->{$elem}} = $elem;
       }
       foreach my $pred (sort{$b <=> $a} keys %REV) {
           push(@sorted, $pred);
       }
    }

    return \@sorted;
}

sub get_word
{
    #description _ returns word
    #@return _ word

    my $rout = shift;  #implicit parameter

    return $rout->{word};
}

sub get_pos
{
    #description _ returns pos
    #@return _ pos

    my $rout = shift;  #implicit parameter

    return $rout->{pos};
}

sub get_pp
{
    #description _ returns pos-prediction hash
    #@return _ pp

    my $rout = shift;  #implicit parameter

    return $rout->{pp};
}

sub get_cols
{
    #description _ returns COLUMNS list
    #@return _ COLS

    my $rout = shift;  #implicit parameter

    return $rout->{COLS};
}

sub set_word
{
    #description _ sets word
    #param1 _ word

    my $rout = shift;  #implicit parameter
    my $word = shift;

    $rout->{word} = $word;
}

sub set_pos
{
    #description _ sets pos
    #param1 _ pos

    my $rout = shift;  #implicit parameter
    my $pos = shift;

    $rout->{pos} = $pos;
}

sub set_pp
{
    #description _ sets pos-prediction hash
    #param1 _ pp

    my $rout = shift;  #implicit parameter
    my $pp = shift;

    $rout->{pp} = $pp;
}

sub set_COLS
{
    #description _ sets COLUMNS list
    #param1 _ COLS

    my $rout = shift;  #implicit parameter
    my $cols = shift;

    $rout->{COLS} = $cols;
}

sub get_pred
{
    #description _ returns prediction for the winner pos
    #@return _ prediction

    my $rout = shift;  #implicit parameter

    return $rout->{pp}->{$rout->{pos}};
}

sub get_pred_2
{
    #description _ returns prediction for a given pos
    #param1 _ pos
    #@return _ prediction

    my $rout = shift;  #implicit parameter
    my $pos = shift;

    return $rout->{pp}->{$pos};
}

sub push_pp
{
    #description _ pushes pos-prediction onto the pp hash
    #param1 _ pos
    #param2 _ prediction

    my $rout = shift;  #implicit parameter
    my $pos = shift;
    my $pred = shift;

    $rout->{pp}->{$pos} = $pred;
}

1;


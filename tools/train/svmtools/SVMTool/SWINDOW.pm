package SWINDOW;

#Copyright (C) 2004 Jesus Gimenez and Lluis Marquez

#This library is free software; you can redistribute it and/or
#modify it under the terms of the GNU Lesser General Public
#License as published by the Free Software Foudation; either
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

# ======================================================================================
# =================================== WINDOW METHODS ===================================
# ======================================================================================

sub new
{
    #description _ creates a new window of a given length
    #
    #              WINDOW --> length of the window
    #                         window itself
    #
    #param1 _ length of the window
    #param2 _ window core word position

    my $class = shift;     #implicit parameter
    my $length = shift;
    my $corepos = shift;

    my $rwindow = { len => $length, core => $corepos, win => [] };

    my $win = $rwindow->{win};

    while ($length > 0) {
       my @elem = (0, $COMMON::emptyword, $COMMON::emptypos, $COMMON::emptypos, $COMMON::emptyword);
       push(@{$win}, \@elem);
       $length--;
    }

    bless $rwindow, $class;

    return $rwindow;
}

sub get_len
{
    #description _ returns the length of the given window
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    return $rwin->{len};
}

sub get_core
{
    #description _ returns the core word position of the given window
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    return $rwin->{core};
}

sub copy
{
    #description _ creates a copy of the given window
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    my $rwindow = { len => $rwin->get_len, core => $rwin->get_core, win => [] };
    my $i = 0;

    my $win = $rwindow->{win};

    while ($i < $rwin->get_len) {
       my @elem = @{$rwin->{win}[$i]};
       push(@{$win}, \@elem);
       $i++;
    }

    bless $rwindow, ref($rwin);

    return $rwindow;
}

sub reset
{
    #description _ resets a given window position
    #param1 _ window reference  (implicit)
    #param2 _ i position where to push the new item

    my $rwin = shift;
    my $i = shift;

    $rwin->{win}[$i]->[0] = 0;
    $rwin->{win}[$i]->[1] = $COMMON::emptyword;
    $rwin->{win}[$i]->[2] = $COMMON::emptypos;
    $rwin->{win}[$i]->[3] = $COMMON::emptypos;
    $rwin->{win}[$i]->[4] = $COMMON::emptyword;
    my $j = 5;
    while ($j < scalar(@{$rwin->{win}[$i]})) {
       $rwin->{win}[$i]->[$j] = $COMMON::emptypos;
       $j++;
    }
}

sub is_empty
{
    #description _ returns true if the i-th position is empty
    #param1 _ window reference  (implicit)
    #param2 _ i position where to push the new item

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$i]->[0] == 0);
}

sub set_cols
{
    #description _ sets extra COLUMNS in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ i position where to push the new item
    #param3 _ columns to push

    my $rwin = shift;
    my $i = shift;
    my $entry = shift;

    if (defined(@{$entry})) {
       my $j = 0;
       while ($j < scalar(@{$entry})) {
          $rwin->{win}[$i]->[5+$j] = $entry->[$j];
          $j++;
       }
    }
}

sub set_word
{
    #description _ sets the WORD in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ i position where to push the new item
    #param3 _ word to push

    my $rwin = shift;
    my $i = shift;
    my $word = shift;

    $rwin->{win}[$i]->[1] = $word;
}

sub set_pos
{
    #description _ sets the POS in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ i position where to push the new item
    #param3 _ pos to push

    my $rwin = shift;
    my $i = shift;
    my $pos = shift;

    $rwin->{win}[$i]->[2] = $pos;
}

sub set_kamb
{
    #description _ sets the AMBIGUITY CLASS in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ i position where to push the new item
    #param3 _ ambiguity class list reference to push

    my $rwin = shift;
    my $i = shift;
    my $kamb = shift;

    $rwin->{win}[$i]->[3] = $kamb;
}

sub set_actual_word
{
    #description _ sets the ACTUAL WORD in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ i position where to push the new item
    #param3 _ word to push

    my $rwin = shift;
    my $i = shift;
    my $word = shift;

    $rwin->{win}[$i]->[4] = $word;
}

sub set_core_word
{
    #description _ sets the WORD in the core position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ word to push

    my $rwin = shift;
    my $word = shift;

    $rwin->{win}[$rwin->get_core]->[1] = $word;
}

sub set_core_pos
{
    #description _ sets the POS in the core position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ pos to push

    my $rwin = shift;
    my $pos = shift;

    $rwin->{win}[$rwin->get_core]->[2] = $pos;
}

sub set_core_actual_word
{
    #description _ sets the ACTUAL WORD in the core position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ word to push

    my $rwin = shift;
    my $word = shift;

    $rwin->{win}[$rwin->get_core]->[4] = $word;
}

sub get_col_relative
{
    #description _ returns the COLUMN in a given position, relative to the window core,
    #              assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ relative item position
    #param3 _ column index

    my $rwin = shift;
    my $i = shift;
    my $column = shift;

    if ($column < 2) {
       return($rwin->{win}[$rwin->get_core+$i]->[1+$column]);
    }
    else {
       return($rwin->{win}[$rwin->get_core+$i]->[3+$column]);
    }
}

sub get_word
{
    #description _ returns the WORD in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$i]->[1]);
}

sub get_word_relative
{
    #description _ returns the WORD in a given position, relative to the window core,
    #              assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ relative item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$rwin->get_core+$i]->[1]);
}

sub get_pos
{
    #description _ returns the POS in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$i]->[2]);
}

sub get_pos_relative
{
    #description _ returns the POS in a given position, relative to the window core,
    #              assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ relative item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$rwin->get_core+$i]->[2]);
}

sub get_kamb
{
    #description _ returns the AMBIGUITY CLASS in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$i]->[3]);
}

sub get_kamb_relative
{
    #description _ returns the AMBIGUITY CLASS in a given position, relative to the window core,
    #              assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ relative item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$rwin->get_core+$i]->[3]);
}

sub get_actual_word
{
    #description _ returns the ACTUAL WORD in a given position, assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$i]->[4]);
}

sub get_actual_word_relative
{
    #description _ returns the ACTUAL WORD in a given position, relative to the window core,
    #              assumed the position is not empty
    #param1 _ window reference  (implicit)
    #param2 _ relative item position

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$rwin->get_core+$i]->[4]);
}

sub get_core_word
{
    #description _ returns the WORD in the core position, assumed the position is not empty
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    return($rwin->{win}[$rwin->get_core]->[1]);
}

sub get_core_pos
{
    #description _ returns the POS in the core position, assumed the position is not empty
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    return($rwin->{win}[$rwin->get_core]->[2]);
}

sub get_core_kamb
{
    #description _ returns the AMBIGUITY CLASS in the core position, assumed the position is not empty
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    return($rwin->{win}[$rwin->get_core]->[3]);
}

sub get_core_actual_word
{
    #description _ returns the ACTUAL WORD in the core position, assumed the position is not empty
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    return($rwin->{win}[$rwin->get_core]->[4]);
}


sub prepare
{
    #description _ resets items according to the text direction either
    #              at the right of the core right-nearest sentence-final (LR)
    #              or at the left of the core left-nearest sentence-final (RL)
    #param1 _ window reference  (implicit)
    #param2 _ text direction  (LR/RL)

    my $rw = shift;
    my $direction = shift;

    my $rwin = $rw->copy();

    my $i = $rwin->get_core;
    my $eos = 0;
    while (($i < $rwin->get_len) and (!$eos)) {
       my $item = $rwin->get_word($i);
       if (COMMON::end_of_sentence($item)) { $eos = 1; }
       $i++;
    }
    if (($direction eq $COMMON::rlmode) and ($eos)) { $i--; }
    while ($i < $rwin->get_len) {
       $rwin->reset($i);
       $i++;
    }

    $i = $rwin->get_core;
    $eos = 0;
    while (($i >= 0) and (!$eos)) {
       my $item = $rwin->get_word($i);
       if (COMMON::end_of_sentence($item)) { $eos = 1; }
       $i--;
    }
    if (($direction eq $COMMON::lrmode) and ($eos)) { $i++; }
    while ($i >= 0) {
       $rwin->reset($i);
       $i--;
    }

    $rwin->set_core_pos($COMMON::emptypos);

    return $rwin;
}

sub active
{
    #description _ returns true if the given window is an active one.
    #              That is, the core item is not empty. (!=$wseparator).
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    return (!($rwin->is_empty($rwin->get_core)));
}

sub lshift
{
    #description _ shift sliding window n positions left
    #param1 _ window reference  (implicit)
    #param2 _ n

    my $rwin = shift;
    my $n = shift;    

    while ($n>0) {
	my $i = 1;
        while ($i < $rwin->get_len) {
           my $j = 0;
           while ($j < scalar(@{$rwin->{win}[$i]})) {
              $rwin->{win}[$i-1]->[$j] = $rwin->{win}[$i]->[$j];
              $j++;
           }
           $i++;
        }
        $rwin->reset($rwin->get_len - 1);
        $n--;
    }
}

sub print
{
    #description _ print onto the standard output raw the window content;
    #param1 _ window reference  (implicit)

    my $rwin = shift;

    my $i = 0;

    while ($i < $rwin->get_len) {
       print STDERR "W", $i - $rwin->get_core, ": \t";
       printf STDERR "%-20s %-20s %-5s %-20s", $rwin->{win}[$i]->[1], "(".$rwin->{win}[$i]->[4].")", $rwin->{win}[$i]->[2], (($rwin->{win}[$i]->[3]) == $COMMON::emptypos ? $rwin->{win}[$i]->[3] : join($COMMON::innerseparator, @{$rwin->{win}[$i]->[3]}));
       my $j = 5;
       while ($j < scalar(@{$rwin->{win}[$i]})) {
          printf STDERR " %-10s", $rwin->{win}[$i]->[$j];
          $j++;
       }
       printf STDERR "\n";
       $i++;
    }
}

sub push
{
    #description _ push a given entry onto the sliding window at the given position
    #              ++ CARD generalization
    #param1 _ window reference  (implicit)
    #param2 _ word to push
    #param3 _ pos to push
    #param4 _ i position where to push the new item
    #param5 _ extra columns (list), e.g {IOB ...}

    my $rwin = shift;
    my $word = shift;
    my $pos = shift;
    my $i = shift;
    my $entry = shift;

    $rwin->{win}[$i]->[3] = $COMMON::emptypos;
    $rwin->{win}[$i]->[4] = $word;

    if ($word =~ /^[0-9]+$/) { $word = "\@CARD"; }
    elsif ($word =~ /^[0-9]+[\.\,\!\?:]+$/) { $word = "\@CARDPUNCT"; }
    elsif ($word =~ /^[0-9]+[:\.\,\/\\\-][0-9\.\,\-\\\/]+$/) { $word = "\@CARDSEPS"; }
    elsif ($word =~ /^[0-9]+[^0-9]+.*$/) { $word = "\@CARDSUFFIX"; }

    $rwin->{win}[$i]->[0] = 1;
    $rwin->{win}[$i]->[1] = $word;
    $rwin->{win}[$i]->[2] = $pos;
 
    if (defined(@{$entry})) {
       my $j = 0;
       while ($j < scalar(@{$entry})) {
          $rwin->{win}[$i]->[5+$j] = $entry->[$j];
          $j++;
       }
    }
}

sub pop
{
    #description _ retrieves the item at the given position
    #param1 _ window reference  (implicit)
    #param2 _ position of the item inside the window

    my $rwin = shift;
    my $i = shift;

    return($rwin->{win}[$i]->[1], $rwin->{win}[$i]->[2]);
}

sub pop_core
{
    #description _ retrieves the item at the core position
    #param1 _ window reference  (implicit)
    
    my $rwin = shift;
  
    return($rwin->{win}[$rwin->get_core]->[1], $rwin->{win}[$rwin->get_core]->[2]);
}


1;

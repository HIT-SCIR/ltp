package ATTGEN;

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
use SVMTool::SWINDOW;
use SVMTool::DICTIONARY;

# ================================== ATTRIB GEN ========================================

# ------------------- ATTGEN ----------------------------------
my $fseparator = ",";
# ----------------- whole sentence features
my $slastw = "Swn";
# -----------------------------------------------------------------------------
my $WMARK = "w";      # words
my $PMARK = "p";      # POS'
my $KMARK = "k";      # ambiguity classes
my $MMARK = "m";      # maybe's
my $MFTMARG = "f";    # SENEN MFT f(-1) --> f-1:NN
my $aMARK = "a";      # prefixes
my $zMARK = "z";      # suffixes
my $caMARK = "ca";    # character [counting from the beginning of the token, starting at 1]
my $czMARK = "cz";    # character [counting from the end of the token, starting at 1]
my $LMARK = "L";      # token length
my $SAMARK = "SA";    # starts with capital letter
my $AAMARK = "AA";    # all upper case
my $NMARK = "SN";     # starts with number
my $saMARK = "sa";    # starts with lower case
my $aaMARK = "aa";    # all lower case
my $CAMARK = "CA";    # contains a capital letter
my $CAAMARK = "CAA";  # contains several capital letters
my $CPMARK = "CP";    # contains period(s)
my $CCMARK = "CC";    # contains comma(s)
my $CNMARK = "CN";    # contains number(s)
my $MWMARK = "MW";    # contains underscores (multi-word)
my $COLMARK = "C";    # contains underscores (multi-word)
my $DOUBLE = "DOU";   #double 新加
my $BUSHOUA = "bsa";    #新加
my $BUSHOUZ = "bsz";    #新加
# -----------------------------------------------------------------------------

sub check_arguments
{
    #description _ returns true if feature arguments must be checked
    #param1 _ feature mark

    my $f = shift;

    return (($f eq $WMARK) or ($f eq $PMARK) or ($f eq $KMARK) or ($f eq $MMARK) or ($f eq $COLMARK));
}


sub push_COLUMN
{
    #description _ pushes a COLUMN feature onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ column index
    #param4 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $column = shift;
    my $args = shift;

    my $i = 0;
    my @ctx;
    my @f;
    while ($i < scalar(@{$args})) {
       push(@f, $rwin->get_col_relative($args->[$i], $column));
       push(@ctx, $args->[$i]);
       $i++;
    }

    $rattribs->{$COLMARK.$column.$COMMON::valseparator.join($fseparator, @ctx)} = join($COMMON::valseparator, @f);
}

sub push_word
{
    #description _ pushes a word feature onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $i = 0;
    my @ctx;
    my @f;
    while ($i < scalar(@{$args})) {
       push(@f, $rwin->get_word_relative($args->[$i]));
       push(@ctx, $args->[$i]);
       $i++;
    }

    $rattribs->{$WMARK.join($fseparator, @ctx)} = join($COMMON::valseparator, @f);
}

sub push_pos
{
    #description _ pushes a pos feature onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $i = 0;
    my @ctx;
    my @f;
    while ($i < scalar(@{$args})) {
       my $pos = $rwin->get_pos_relative($args->[$i]);
       if ($pos eq $COMMON::emptypos) {
          my $word = $rwin->get_word_relative($args->[$i]);
          if ($word eq $COMMON::emptyword) {
	     push(@f, $COMMON::emptypos);
	  }
	  else {
             my $rpotser = $rwin->get_kamb_relative($args->[$i]); 
             if (defined(@{$rpotser})) { push(@f, join($COMMON::innerseparator, @{$rpotser})); } 
  	     else { push(@f, $COMMON::unkamb); }
	  }
       }
       else { push(@f, $pos); }
       push(@ctx, $args->[$i]);
       $i++;
    }

    $rattribs->{$PMARK.join($fseparator, @ctx)} = join($COMMON::valseparator, @f);
}

sub push_kamb
{
    #description _ pushes an ambiguity_class feature onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference (not n-grams)

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_word_relative($args->[0]);

    if ($word eq $COMMON::emptyword) {
       $rattribs->{$KMARK.$args->[0]} = $COMMON::emptypos;
    }
    else {
       my $rpotser = $rwin->get_kamb_relative($args->[0]); 
       if (defined(@{$rpotser})) {
          $rattribs->{$KMARK.$args->[0]} = join($COMMON::valseparator, @{$rpotser});
       }
       else {
          $rattribs->{$KMARK.$args->[0]} = $COMMON::unkamb;
       }
    }
}

sub push_maybe
{
    #description _ pushes a maybe feature onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference (not n-grams)

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_word_relative($args->[0]);

    if ($word eq $COMMON::emptyword) {
       $rattribs->{$MMARK.$args->[0].$COMMON::valseparator.$COMMON::emptypos} = 1;
    }
    else {
       my $rpotser = $rwin->get_kamb_relative($args->[0]); 
       if (defined(@{$rpotser})) {     
          foreach my $ps (@{$rpotser}) {
             $rattribs->{$MMARK.$args->[0].$COMMON::valseparator.$ps} = 1;       #POTSER
          }
       }
       else {
          $rattribs->{$MMARK.$args->[0].$COMMON::valseparator.$COMMON::unkamb} = 1;
       }
    }
}

#改动
sub push_prefix
{
    #description _ pushes a prefix feature onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();
    my $n = $args->[0];
    my $len = length($word);

    if ($len >= $n) { $rattribs->{$aMARK.$n} = substr($word, 0, $n); }
    else {
       $rattribs->{$aMARK.$n} = substr($word, 0, $len);
       my $i = $len;
       while ($i < $n) { $rattribs->{$aMARK.$n} .= $COMMON::valseparator; $i++; }
    }
}

#改动
sub push_suffix
{
    #description _ pushes a suffix feature onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();
    my $n = $args->[0];
    my $len = length($word);

    if ($len >= $n) { $rattribs->{$zMARK.$n} = substr($word, $len-$n, $n); }
    else {
       my $i = $len;
       while ($i < $n) { $rattribs->{$zMARK.$n} .= $COMMON::valseparator; $i++; }
       $rattribs->{$zMARK.$n} .= substr($word, 0, $len);
    }
}

sub push_ca
{
    #description _ pushes a ca feature [character, counting from the beginning] onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();
    my $n = $args->[0];
    my $len = length($word);

    if ($len >= $n) { $rattribs->{$caMARK.$n} = substr($word, $n-1, 1); }
    else { $rattribs->{$caMARK.$n} = $COMMON::valseparator; }
}

sub push_cz
{
    #description _ pushes a ca feature [character, counting from the end] onto a feature hash
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();
    my $n = $args->[0];
    my $len = length($word);

    if ($len >= $n) { $rattribs->{$czMARK.$n} = substr($word, $len-$n, 1); }
    else { $rattribs->{$czMARK.$n} = $COMMON::valseparator; }
}

sub push_length
{
    #description _ pushes a length feature [word length]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();
    my $len = length($word);

    $rattribs->{$LMARK} = $len;
}

sub push_SA
{
    #description _ pushes an A feature [starts with capital letter]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^[A-Z].*$/) { $rattribs->{$SAMARK} = 1; }
}

sub push_AA
{
    #description _ pushes an AA feature [all upper case]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^[A-Z]+$/) { $rattribs->{$AAMARK} = 1; }
}

sub push_N
{
    #description _ pushes an N feature [starts with number]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^[0-9].*$/) { $rattribs->{$NMARK} = 1; }
}

sub push_CN
{
    #description _ pushes a CN feature [contains a number]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^.*[0-9].*$/) { $rattribs->{$CNMARK} = 1; }
}


sub push_sa
{
    #description _ pushes a sa feature [starts with lower case]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^[a-z].*$/) { $rattribs->{$saMARK} = 1; }
}

sub push_aa
{
    #description _ pushes an aa feature [all lower case]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^[a-z]+$/) { $rattribs->{$aaMARK} = 1; }
}

sub push_CA
{
    #description _ pushes a CA feature [contains Capital Letter]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^.+[A-Z].*$/) { $rattribs->{$CAMARK} = 1; }
}

sub push_CAA
{
    #description _ pushes a CAA feature [contains several Capital Letters]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^.*[A-Z].*[A-Z].*$/) { $rattribs->{$CAAMARK} = 1; }
}

sub push_CP
{
    #description _ pushes a CP feature [contains a period]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^.*[\.].*$/) { $rattribs->{$CPMARK} = 1; }
}

sub push_CC
{
    #description _ pushes a CC feature [contains a comma]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^.*[\,].*$/) { $rattribs->{$CCMARK} = 1; }
}

sub push_MW
{
    #description _ pushes a CC feature [contains a comma]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();

    if ($word =~ /^.*[\-].*$/) { $rattribs->{$MWMARK} = 1; }
}

sub get_core_word_from_hash
{
    #description _ returns the feature hash core word
    #param1  _ feature hash reference
    #@return _ core word (w0:__XXX___)

    my $rattribs = shift;

    return $rattribs->{$WMARK."0"};
}

sub get_core_word_from_list
{
    #description _ returns the feature list core word
    #param1  _ feature list reference
    #@return _ core word (w0:__XXX___)

    my $lattribs = shift;

    my $len = @{$lattribs};
    my $i = 1;
    my $stop = 0;
    my $item;

    my $WT = $WMARK."0:";
    while (($i < $len) and (!$stop)) {
       $item = $lattribs->[$i];
       chomp($item);
       if ($item =~ /$WT/) {
	  $item =~ s/$WT//g;
	  $stop = 1;
       }
       else { $i++; }
    }

    return $item;
}

#新加


#新加
sub push_BSA
{
	my $rattribs = shift;
  my $rwin = shift;
  my $args = shift;
  my $Unihan = shift;
  my $BS = shift;

  my $word = $rwin->get_core_word();
  my $len = length($word);
  my $n = $args->[0];
  
  my @fs;
  my $innum;
  
  if($len >= $n * 2)
  {
  	my $k = 0;
  	while($k < $n)	
  	{
  		my $key = substr($word, $k*2 ,2);
  		my $ind = 0;
  		if(exists $Unihan->{$key})
  		{
  			$ind = oct($Unihan->{$key});
  		}
  		my $i = 0;
			foreach my $in (@{$BS})
			{
				$innum = oct($in);
				if( $ind < $innum )
				{ 
					my $j = $i - 1 ;
					push(@fs,$j);
					last;	
				}
				elsif($ind == $innum)
				{
					my $j = $i ;
					push(@fs,$j);
					last;
				}
				$i++;
			}
			$k++;
  	}
  }
	else
	{
		$n = $len / 2;
		my $k = 0;
  	while($k < $n)	
  	{
  		my $key = substr($word, $k*2 ,2);
  		my $ind = 0;
  		if(exists $Unihan->{$key})
  		{
  			$ind = oct($Unihan->{$key});
  		}
  		my $i = 0;
			foreach my $in (@{$BS})
			{
				$innum = oct($in);
				if( $ind < $innum )
				{ 
					my $j = $i - 1 ;
					push(@fs,$j);
					last;	
				}
				elsif($ind == $innum)
				{
					my $j = $i ;
					push(@fs,$j);
					last;
				}
				$i++;
			}
			$k++;
  	}
	}
  $rattribs->{$BUSHOUA.$n} = join($COMMON::valseparator, @fs);
	
}

sub push_BSZ
{
	my $rattribs = shift;
  my $rwin = shift;
  my $args = shift;
  my $Unihan = shift;
  my $BS = shift;

  my $word = $rwin->get_core_word();
  my $len = length($word);
  my $n = $args->[0];
  
  my @fs;
  my $innum;
  if($len >= $n * 2)
  {
  	my $k = $n;
  	while($k > 0)	
  	{
  		my $key = substr($word, $len - $k*2,2);
  		my $ind = 0;
  		if(exists $Unihan->{$key})
  		{
  			$ind = oct($Unihan->{$key});
  		}
  		my $i = 0;
			foreach my $in (@{$BS})
			{
				$innum = oct($in);
				if( $ind < $innum )
				{ 
					my $j = $i - 1 ;
					push(@fs,$j);
					last;	
				}
				elsif($ind == $innum)
				{
					my $j = $i ;
					push(@fs,$j);
					last;
				}
				$i++;
			}
			$k -= 1;
  	}
  }
	else
	{
		$n = $len / 2;
		my $k = $n;
  	while($k > 0)	
  	{
  		my $key = substr($word, $len - $k*2,2);
  		my $ind = 0;
  		if(exists $Unihan->{$key})
  		{
  			$ind = oct($Unihan->{$key});
  		}
  		my $i = 0;
			foreach my $in (@{$BS})
			{
				$innum = oct($in);
				if( $ind < $innum )
				{ 
					my $j = $i - 1 ;
					push(@fs,$j);
					last;	
				}
				elsif($ind == $innum)
				{
					my $j = $i ;
					push(@fs,$j);
					last;
				}
				$i++;
			}
			$k -= 1;
  	}
	}
  $rattribs->{$BUSHOUZ.$n} = join($COMMON::valseparator, @fs);
	
}

#新加
sub push_DOU
{
		#description _ pushes a CAA feature [contains several Capital Letters]
    #param1 _ feature hash
    #param2 _ sliding window object
    #param3 _ argument list reference

    my $rattribs = shift;
    my $rwin = shift;
    my $args = shift;

    my $word = $rwin->get_core_word();
    my $len = length($word);
		my $NULL = "NULL"; #double 新加
		#if($len == 4)
		#{
		#	my $first = substr($word,0,2);
		#	my $second = substr($word,2,2);
		#	if($first eq $second)
		#	{
		#		$rattribs->{$DOUBLE} = $first;
		#	}
		#}
		#elsif($len == 6)
		#{
		#	my $first = substr($word ,0,2);
		#	my $second = substr($word,4,2);	
		#	if($first eq $second)
		#	{
		#		$rattribs->{$DOUBLE} = $first;
		#	}
		#}
		#elsif($len == 8)
		#{
		#	my $first = substr($word ,0,2);
		#	my $second = substr($word,2,2);
		#	my $third = substr($word,4,2);
		#	my $fourth = substr($word ,6,2);
		#	if(($first eq $second) and ($third eq $ fourth))
		#	{
		#		$rattribs->{$DOUBLE} = $first.$third;
		#	}
		#	elsif(($first eq $third) and ($second eq $ fourth))
		#	{
		#		$rattribs->{$DOUBLE} = $first.$second;
		#	}
	#	}
		
		if($len == 6)
		{
			my $first = substr($word ,0,2);
			my $second = substr($word,2,2);	
			if($first eq $second)
			{
				$rattribs->{$DOUBLE} = $first;
			}
			else
			{
				$rattribs->{$DOUBLE} = $NULL;
			}
		}
		elsif($len == 8)
		{
			my $first = substr($word ,0,2);
			my $second = substr($word,2,2);
			my $third = substr($word,4,2);
			my $fourth = substr($word ,6,2);
			if(($first eq $second) and ($third eq $fourth))
			{
				$rattribs->{$DOUBLE} = $first.$third;
			}
			elsif(($first eq $third) and ($second eq $fourth))
			{
				$rattribs->{$DOUBLE} = $first.$second;
			}
			else
			{
				$rattribs->{$DOUBLE} = $NULL;
			}
		}
		else
		{
				$rattribs->{$DOUBLE} = $NULL;
		}
}
# -----------------------------------------------------------------------------


sub do_features
{
    #description _ responsible for generating a hash containing the attributes
    #              corresponding to the given window.
    #
    #              -> + bigrams + trigrams    (word, pos)
    #
    #param1 _ window reference
    #param2 _ dictionary object reference
    #param3 _ sentence general information list --> (last word)
    #param4 _ feature set list reference

    my $rwin = shift;
    my $rdict = shift;
    my $sinfo = shift;
    my $fs = shift;
    my $Unihan = shift;
    my $BS = shift;

    my $wlength = $rwin->get_len();
    my $wcorepos = $rwin->get_core();
    my %attribs;

    #loading ambiguity classes
    my $i = 0;
    while ($i < $wlength) {
       $rwin->set_kamb($i, $rdict->get_potser($rwin->get_word($i)));
       $i++;
    }

    # whole sentence features
    $attribs{$slastw} = $sinfo->[0];

    foreach my $f (@{$fs}) {
       if ($f->[0] eq $WMARK) { push_word(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $PMARK) { push_pos(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $KMARK) { push_kamb(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $MMARK) { push_maybe(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $aMARK) { push_prefix(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $zMARK) { push_suffix(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $caMARK) { push_ca(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $czMARK) { push_cz(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $LMARK) { push_length(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $NMARK) { push_N(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $SAMARK) { push_SA(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $AAMARK) { push_AA(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $saMARK) { push_sa(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $aaMARK) { push_aa(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $CAMARK) { push_CA(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $CAAMARK) { push_CAA(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $CPMARK) { push_CP(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $CCMARK) { push_CC(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $CNMARK) { push_CN(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $MWMARK) { push_MW(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $DOUBLE) { push_DOU(\%attribs, $rwin, $f->[1]); }
       elsif ($f->[0] eq $BUSHOUA) { push_BSA(\%attribs, $rwin, $f->[1],$Unihan,$BS); }
       elsif ($f->[0] eq $BUSHOUZ) { push_BSZ(\%attribs, $rwin, $f->[1],$Unihan,$BS); }
       elsif ($f->[0] eq $COLMARK) { ### MULTIPLE-COLUMNS <-----
          push_COLUMN(\%attribs, $rwin, $f->[1], $f->[2]);
       }
       else { print STDERR "UNKNOWN FEATURE TYPE! [", $f->[0], "]\n"; }
    }

    return \%attribs;
}

sub generate_features
{
    #description _ responsible for generating a hash containing the features
    #              corresponding to the given window.
    #
    #              -> + bigrams + trigrams    (word, pos)
    #
    #param1  _ window reference
    #param2  _ dictionary object reference
    #param3  _ sentence general information list --> (last word --> ',' '?' '!')
    #param4  _ mode
    #          (generate_features 0-ambiguous-right :: 1-unambiguous-right :: 2-no-right)
    #          (                  3-unsupervised :: 4-unknown words on training...)    
    #param5  _ feature set list reference
    #@return _ attribute hash reference

    my $rwin = shift;
    my $rdict = shift;
    my $sinfo = shift;
    my $mode = shift;
    my $fs = shift;
    my $Unihan = shift;
    my $BS = shift;

    if ($mode != $COMMON::mode1) { #pos information for unseen words is not available
       my $i = $rwin->get_core;
       while ($i < $rwin->get_len) {
          $rwin->set_pos($i, $COMMON::emptypos);
          $i++;
       }
    }

    return do_features($rwin, $rdict, $sinfo, $fs,$Unihan,$BS);# have modified  add $Unihan
}

1;

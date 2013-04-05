package LEMMATIZER;

use SVMTool::COMMON;

# ------------------------------------- SUBS ------------------------------------------

sub LT_load
{
    #description _ loads a dictionary
    #param1  _ dictionary filename
    #param2  _ verbose
    #@return _ dictionary hash ref

    my $dict = shift;
    my $verbose = shift;

    my $FDICT = new IO::File("< $dict") or die "Couldn't open dictionary file: $dict\n";

    my %DICT;
    my $iter;
    while (defined (my $line = <$FDICT>)) {
        #abandoned VBD abandon
        #print $line;
        chomp ($line);
        my @entry = split(" ", $line);
        $DICT{$entry[0].$COMMON::out_valseparator.$entry[1]} = $entry[2];
        $iter++;
        if ($verbose) {
           if (($iter%10000) == 0) { print STDERR "."; }
           if (($iter%100000) == 0) { print STDERR "$iter"; }
        }
    }

    $FDICT->close();

    if ($verbose) { print STDERR "...$iter forms [DONE]\n"; }
    
    return \%DICT;
}

sub LT_tag
{
   #description _ given a word/pos pair returns the lemma according to the given dictionary
   #param1 _ dictionary
   #param2 _ word
   #param3 _ pos

   my $dict = shift;
   my $word = shift;
   my $pos = shift;

   if (exists($dict->{$word.$COMMON::out_valseparator.$pos})) { return $dict->{$word.$COMMON::out_valseparator.$pos}; }
   else {
      if (exists($dict->{lc($word).$COMMON::out_valseparator.$pos})) { return $dict->{lc($word).$COMMON::out_valseparator.$pos}; }
      else { return $word; }
   }
}

1;

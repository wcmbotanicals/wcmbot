# Ralph Tasks
- [x] Make a table of matching scores (not including grid centre weighting), and a separate table of grid centre scores, for each piece tested in test_matcher_functional(). Save
to tmp/matching_score_table.csv
- [x] Investigate whether or not the score table gives clear signals, either in the match score or the grid score, in the pieces not placed correctly in
'test_multipiece_many_pieces_batch()' that could be used to identify failed matches.
- [x] If (AND ONLY IF) the grid centre score is notably lower in the failing pieces, 
consider replacing the current grid centre weighting with a system where matching scores
are not weighted, but potential matches with grid centre scores below a threshold are
rejected.
- [x] Identify a matching score threshold (with or without grid centre score weighting
as appropriate) that is indicative of a possible failed match.
- [ ] Using the currently failing pieces in test_multipiece_many_pieces_batch() as a
guide, consider options for finding better matches when the initial best match
score is below the identified threshold. Possible options include: (1) adding rotations
of (+/- 2.5 degrees and +/- 5 degrees) to the piece (post auto-alignment) and re-running 
matching; (2) removing a small proportion of pixels around the mask edge to eliminate possible shadow artifacts, and re-running matching; (3) considering a wider range of
candidate scales within the existing scale range; (4) other options as identified.

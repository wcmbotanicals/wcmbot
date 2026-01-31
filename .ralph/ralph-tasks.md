# Ralph Tasks
- [x] Add a test to test_matcher_functional.py that tests multipiece matching on
media/pieces/grass_puzzle/difficult_multipiece.jpg. I have saved the true positions
to a temp file at tmp/correct_positions.txt, which you should copy into the test
(don't read from the file at runtime). Note that there two pieces with currently
unknown locations (marked with a ? in the file). The test should assert that all
known pieces are matched correctly. This test is expected to fail currently.
The goal is to have this test pass eventually, but for now it is just to help
track progress.
- [x] Look into the matching score with/without grid centre weight, as well as the grid
centre score component, for the difficult_multipiece.jpg test as well as the existing
test_multipiece_many_pieces_batch() test. By comparing between pieces known to currently
be placed correctly and incorrectly, consider whether there is a signal in any of these scores indicating a high likelihood of a failed match. Based on this, consider
modifying LOW_SCORE_THRESHOLD in matcher.py and/or adding an additional threshold
based on grid centre score.
- [x] Evaluate whether we can improve the number of correct matches on the new test,
as well as the existing test test_multipiece_many_pieces_batch(), by adjusting
some/all of LOW_SCORE_ROTATIONS, LOW_SCORE_MASK_EDGE_FRAC and LOW_SCORE_SCALE_SAMPLES in matcher.py. Ideally we would make the new test pass, but this is not strictly required
at this stage (other tests must still pass). Report any improvements in number of
correct matches.
- [x] Add new variants of the difficult_multipiece.jpg test, as well as the existing
test_multipiece_many_pieces_batch() test, that use mask_mode="ai" for background
removal of individual pieces (not the initial stage of separating into individual
pieces - see similar logic in the app). These should check for the same expected
positions, and should currently fail.
- [ ] Investigate whether using a higher value of LOW_SCORE_MASK_EDGE_FRAC for "ai"
mask mode ONLY (or similar logic to remove edge artifacts typically seen with
AI-based background removal) can improve the number of correct matches on the new tests.
Ideally we would make the new tests pass, but this is not strictly required
at this stage (other tests must still pass). Report any improvements in number of
correct matches.

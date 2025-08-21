PY ?= python3
L ?= 12
TSTEPS ?= 48
SIDE ?= x

.PHONY: install run-sim plot-sim run-entropy plot-entropy clean

install:
	$(PY) -m pip install -r requirements.txt

run-sim:
	$(PY) src/sim_spin2.py --L $(L) --tsteps $(TSTEPS)

plot-sim:
	$(PY) src/sim_spin2.py --L $(L) --tsteps $(TSTEPS) --plot

run-entropy:
	$(PY) src/entropy_cut.py --L $(L) --side $(SIDE)

plot-entropy:
	$(PY) src/entropy_cut.py --L $(L) --side $(SIDE) --plot

clean:
	rm -f figs/*.png

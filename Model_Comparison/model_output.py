class SIRModelOutput:
    """
    Minimal container for S/E/I/R trajectories and time vector.
    Provides numpy-like slices for downstream plotting and assertions.
    """

    def __init__(self, t, S, I, R):
        self.t = t
        self.S = S

        self.I = I
        self.R = R

    def as_dict(self):
        return {"t": self.t, "S": self.S, "I": self.I, "R": self.R}

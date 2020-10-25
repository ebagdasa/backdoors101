from backdoors.backdoor import Backdoor


class PatternBackdoor(Backdoor):

    def apply_backdoor(self, batch, i=-1):
        raise NotImplemented
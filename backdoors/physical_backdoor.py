from backdoors.backdoor import Backdoor


class PhysicalBackdoor(Backdoor):

    def apply_backdoor(self, batch, i=-1):
        raise NotImplemented
from synthesizers.synthesizer import Synthesizer


class PhysicalSynthesizer(Synthesizer):

    def apply_backdoor(self, batch, i=-1):
        raise NotImplemented
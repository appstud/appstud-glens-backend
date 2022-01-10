import pdb
import logging



class options:
    
    def __init__(self):
        self.GET_AGE_SEX_GLASSES=False
        self.GET_HAIR_COLOR=False
        self.GET_MASK=False
        self.GET_POSE=False
        self.GET_FACE_RECO=False

    def reset(self):
        self.GET_AGE_SEX_GLASSES=False
        self.GET_HAIR_COLOR=False
        self.GET_MASK=False
        self.GET_POSE=False
        self.GET_FACE_RECO=False
    
    def __str__(self): 
        return ', '.join("%s: %s" % item for item in self.__dict__.items())
    
    __repr__ = __str__


def decode_pipeline(currentChannel,s):
    global opts
    components=list(map(lambda x:x.strip().split(" ") ,s.split("|")))
        
    for i,comp in enumerate(components):
        if(comp[0]==currentChannel and len(components)!=i+1):
            destinationChannel=components[i+1][0]
            for attr in comp[1:]:
                field,value=attr.split("=")
                if(hasattr(opts,field)):
                    setattr(opts,field,value)
                else:
                    logging.warning("field {} does not exist".format(field))
            break
        elif(len(components)==i+1):
            destinationChannel=components[0][0]
    logging.debug("decoding pipeline: current channel: {}, destination channel: {}, opts: {}".format(currentChannel,destinationChannel,opts))


if(__name__=="__main__"):
    
    opts=options()
    pipeline="results | face-detection | tensorflow-calls GET_HAIR_COLOR=True | covid19_clustering"
    
    currentChannel=["face-detection","tensorflow-calls","covid19_clustering"]
    for ch in currentChannel:
        decode_pipeline(ch,pipeline)

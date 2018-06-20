
#class Osprey(BaseEstimator):
#    """
#    Overrides scikit-learn calls to make inheritance work in the 
#    Osprey bayesian optimisation package.
#    """
#
#    def __init__(self, **kwargs):
#        pass
#
#    def get_params(self, deep = True):
#        """
#        Hack that overrides the get_params routine of BaseEstimator.
#        self.get_params() returns the input parameters of __init__. However it doesn't
#        handle inheritance well, as we would like to include the input parameters to
#        __init__ of all the parents as well.
#
#        """
#
#        # First get the name of the class self belongs to
#        base_class = self.__class__.__base__
#
#        # Then get names of the parents and their parents etc
#        # excluding 'object'
#        parent_classes = [c for c in base_class.__bases__ if c.__name__ not in "object"]
#        # Keep track of whether new classes are added to parent_classes
#        n = 0
#        n_update = len(parent_classes)
#        # limit to 10 generations to avoid infinite loops
#        for i in range(10):
#            for parent_class in parent_classes[n:]:
#                parent_classes.extend([c for c in parent_class.__bases__ if 
#                    (c.__name__ not in "object" and c not in parent_classes)])
#            n = n_update
#            n_update = len(parent_classes)
#            if n == n_update:
#                break
#        else:
#            print("Warning: Only included first 10 generations of parents of the called class")
#
#        params = BaseEstimator.get_params(self)
#        for parent in names_of_parents:
#            parent_init = (parent + ".__init__")
#
#            # Modified from the scikit-learn BaseEstimator class
#            parent_init_signature = signature(parent_init)
#            for p in (p for p in parent_init_signature.parameters.values() 
#                    if p.name != 'self' and p.kind != p.VAR_KEYWORD):
#                if p.name in params:
#                    raise InputError('This should never happen')
#                if hasattr(self, p.name):
#                    params[p.name] = getattr(self, p.name)
#                else:
#                    params[p.name] = p.default
#
#        return params
#
#    def set_params(self, **params):
#        """
#        Hack that overrides the set_params routine of BaseEstimator.
#
#        """
#        for key, value in params.items():
#            key, delim, sub_key = key.partition('__')
#
#            if delim:
#                nested_params[key][sub_key] = value
#            else:
#                setattr(self, key, value)
#
#        return self
#
#    def score(self, x, y = None):
#        # Osprey maximises a score per default, so return minus mae/rmsd and plus r2
#        if self.scoring_function == "r2":
#            return self._score(x, y)
#        else:
#            return - self._score(x, y)

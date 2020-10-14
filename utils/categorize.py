import pandas as pd

from pandas.api.types import is_numeric_dtype


class Categorize(object):
    def __init__(
        self,
        data,
        binary_cols=None,
        hierarchical_continuous_cols=None,
        n_hierarchical_bins=10,
        non_hierarchical_cols=None,
        max_non_hierarchical_classes=10
    ):
        self.data = data
        self.binary_cols = binary_cols
        self.hierarchical_continuous_cols = hierarchical_continuous_cols
        self.n_hierarchical_bins = n_hierarchical_bins
        self.non_hierarchical_cols = non_hierarchical_cols
        self.max_non_hierarchical_classes = max_non_hierarchical_classes

    def _transform_binary(self):
        """
            This function maps the values from a two-valued field to {0, 1}. The values
            are ordered so we are able to discover which number corresponds to each value
        """
        if self.binary_cols is None:
            return
        
        for col in self.binary_cols:
            dict_values = {
                value: i for i, value in enumerate(sorted(self.data[col].unique()))
            }

            assert len(dict_values) == 2, f"Informed Column '{col}' is Not Binary!"
            
            self.data[col] = self.data[col].map(dict_values)
    
    def _transform_hierarchical_continuous(self):
        """
            This function creates bins to either continuous or discrete numeric values
        """
        if self.hierarchical_continuous_cols is None:
            return
        
        # Should I deal with outliers differently?
        for col in self.hierarchical_continuous_cols:
            self.data[col] = pd.to_numeric(self.data[col])
            self.data[col] = pd.cut(self.data[col], bins=self.n_hierarchical_bins, labels=range(self.n_hierarchical_bins))
    
    def _transform_non_hierarchical(self):
        """
            This function creates an indicator column to each one of the values in
            the original column, i.e places a 1 when the record presents a value.
            Note that this function considers only the 'max_non_hierarchical_classes' most
            important values and gathers all the remaining values in the group 'others'
        """
        if self.non_hierarchical_cols is None:
            return
        
        for col in self.non_hierarchical_cols:
            if is_numeric_dtype(self.data[col]):
                self.data[col] = self.data[col].astype(str)
            
            main_values = self.data[col].value_counts()[:self.max_non_hierarchical_classes].index
            self.data.loc[~self.data[col].isin(main_values), col] = "others"
            
            self.data[col] = self.data[col].astype(str)
            self.data[col] = self.data[col].str.lower()
            self.data[col] = self.data[col].str.strip()

            for value in self.data[col].unique():
                new_name = f"{col}_{value}"
                self.data[new_name] = 0
                self.data.loc[self.data[col] == value, new_name] = 1
            
            self.data = self.data.drop(col, axis=1)
            
    
    def transform_data(self) -> pd.DataFrame:
        self._transform_binary()
        self._transform_hierarchical_continuous()
        self._transform_non_hierarchical()
        
        return self.data

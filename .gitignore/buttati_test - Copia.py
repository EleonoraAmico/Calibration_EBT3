# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:34:47 2025

@author: Ele_p
"""

# class TestGeneralizedPolynomial:
#     def setup_method(self):
#         self.fitter = CurveFitter()

#     def test_zero_input(self):
#         """
#         GIVEN: Zero input with various parameters
#         WHEN: The polynomial scaling function is applied
#         THEN: The output should be zero
#         """
#         assert self.fitter._generalized_polynomial(0, 1, 1, 5) == 0
#         assert self.fitter._generalized_polynomial(0, -1, 5, 5) == 0
#         assert self.fitter._generalized_polynomial(0, 10, -3, 5) == 0

#     def test_unity_scaling(self):
#         """
#         GIVEN: Input x=1 with various parameters
#         WHEN: The polynomial scaling function is applied
#         THEN: The output should be the sum of parameters a and b
#         """
#         assert self.fitter._generalized_polynomial(1, 2, 3, 5) == 5  # 2*1 + 3*1^2
#         assert self.fitter._generalized_polynomial(1, -1, 1, 5) == 0  # -1*1 + 1*1^3

#     @given(x=st.floats(min_value=-1e3, max_value=1e3),
#            a=st.floats(min_value=-1e2, max_value=1e2),
#            b=st.floats(min_value=-1e2, max_value=1e2))
#     @settings(suppress_health_check=[HealthCheck.differing_executors])
#     def test_linear_case(self, x, a, b):
#         """
#         GIVEN: Random inputs with r=1
#         WHEN: The polynomial scaling function is applied
#         THEN: The function should behave linearly with slope (a+b)
#         """
#         assume(not np.isnan(x) and not np.isnan(a) and not np.isnan(b))
#         result = self.fitter._generalized_polynomial(x, a, b, 1)
#         expected = x * (a + b)
#         assert np.abs(result - expected) < 1e-10

#     @given(x=st.floats(min_value=0, max_value=1e7),
#            a=st.floats(min_value=-1e2, max_value=1e2),
#            b=st.floats(min_value=-1e2, max_value=1e2))
#     @settings(suppress_health_check=[HealthCheck.differing_executors])
#     def test_parabolic_symmetry(self, x, a, b):
#         """
#         GIVEN: Random positive inputs with r=2
#         WHEN: The polynomial scaling function is evaluated at x and -x
#         THEN: The function should display parabolic symmetry properties:
#               - If a=0: f(-x) = f(x) (symmetric about y-axis)
#               - If b=0: f(-x) = -f(x) (antisymmetric about origin)
#               - In general: f(-x) = (-a)x + b*x^2
#         """
#         assume(not np.isnan(x) and not np.isnan(a) and not np.isnan(b))
        
#         # Evaluate function at x and -x
#         f_x = self.fitter._generalized_polynomial(x, a, b, 2)
#         f_minus_x = self.fitter._generalized_polynomial(-x, a, b, 2)
        
#         # Test symmetry properties
#         if a == 0:  # When a = 0
#             # Pure quadratic should be symmetric about y-axis
#             assert np.abs(f_x - f_minus_x) < 1e-10
#         elif b == 0:  # When b = 0
#             # Pure linear should be antisymmetric about origin
#             assert np.abs(f_x + f_minus_x) < 1e-10
#         else:
#             # General case: f(-x) = -ax + bx^2
#             expected_symmetry = -a*x + b*x**2
#             assert np.abs(f_minus_x - expected_symmetry) < 1e-10

#     @given(x=st.floats(min_value=0, max_value=65535).filter(lambda x: x != 0),
#            a=st.floats(min_value=-1e2, max_value=1e2),
#            b=st.floats(min_value=-1e2, max_value=1e2),
#            r=st.floats(min_value=5, max_value=10))
#     @settings(suppress_health_check=[HealthCheck.differing_executors])
#     def test_polynomial_case(self, x, a, b, r):
#         """
#         GIVEN: Random inputs
#         WHEN: The polynomial scaling function is applied
#         THEN:  The output should be correctly computed for each element
#         """
#         assume(not np.isnan(x) and not np.isnan(a) and not np.isnan(b) and not np.isnan(r))
#         result = self.fitter._generalized_polynomial(x, a, b, r)
#         expected = a*x + b*x**r
#         assert np.abs(result - expected) < 1e-10
    
#     def test_polynomial_warning_negative_r(self):
#         """Test warning generation for zero values with negative power
        
#         GIVEN: Input array containing zero and negative power r
#         WHEN: The generalized polynomial function is called  
#         THEN: A warning is raised for the zero value positions
#         """
#         x = np.array([1.0, 0.0, 3.0])
#         a, b = 2.0, 3.0
#         r = -2.0
        
#         with pytest.warns(Warning) as warning_info:
#             result = self.fitter._generalized_polynomial(x, a, b, r)

#         assert len(warning_info) == 1
#         assert "Invalid values found at x positions: [1]" in str(warning_info[0].message)

#     def test_polynomial_calculation_negative_r(self):
#         """Test polynomial calculation with negative power on valid inputs
        
#         GIVEN: Non-zero input values and negative power r
#         WHEN: The generalized polynomial function is called
#         THEN: The function correctly computes a*x + b*x^r
#         """
#         x = np.array([1.0, 2.0, 3.0])
#         a, b = 2.0, 3.0
#         r = -2.0
        
#         # Expected result: 2x + 3x^(-2)
#         expected = 2.0 * x + 3.0 * x**(-2.0)
        
#         result = self.fitter._generalized_polynomial(x, a, b, r)
        
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_polynomial_array_shape(self):
#         """Test output shape matches input shape
        
#         GIVEN: Input array of specific shape
#         WHEN: The generalized polynomial function is called
#         THEN: The output array has the same shape as input
#         """
#         x = np.array([[1.0, 2.0], [3.0, 4.0]])
#         a, b = 2.0, 3.0
#         r = 5.5
        
#         result = self.fitter._generalized_polynomial(x, a, b, r)
        
#         assert result.shape == x.shape

#     def test_array_input(self):
#         """
#         GIVEN: Array input
#         WHEN: The polynomial scaling function is applied
#         THEN: The output should be correctly computed for each element
#         """
#         x = np.array([0, 1, 2])
#         result = self.fitter._generalized_polynomial(x, 1, 2, 5)
#         expected = np.array([0, 3, 66])  # [1*0 + 2*0^5, 1*1 + 2*1^5, 1*2 + 2*2^5]
#         np.testing.assert_array_almost_equal(result, expected)

#     @pytest.mark.parametrize("x,a,b,r,expected", [
#         (2, 1, 1, 2, 6),    # 1*2 + 1*2^2
#         (3, 2, -1, 2, -3),  # 2*3 + (-1)*3^2
#         (-1, 1, 1, 3, -2)   # 1*(-1) + 1*(-1)^3
#     ])
#     def test_specific_values(self, x, a, b, r, expected):
#         """
#         GIVEN: Specific input values and parameters
#         WHEN: The polynomial scaling function is applied
#         THEN: The output should match pre-calculated results
#         """
#         result = self.fitter._generalized_polynomial(x, a, b, r)
#         assert abs(result - expected) < 1e-10

#     def test_zero_input_non_negative_r(self):
#         """
#         GIVEN: Zero input with non-negative power r
#         WHEN: The polynomial scaling function is applied
#         THEN: The output should be zero
#         """
#         x = np.array([0, 0, 0])
#         assert np.all(self.fitter._generalized_polynomial(x, 1, 2, 2) == 0)
#         assert np.all(self.fitter._generalized_polynomial(x, 1, 2, 0) == 2)
#         assert np.all(self.fitter._generalized_polynomial(x, 1, 2, 1) == 0)

#     def test_extreme_value_handling(self):
#         """
#         GIVEN: Extreme input values
#         WHEN: The polynomial scaling function is applied
#         THEN: The function handles large/small values without unexpected behavior
#         """
#         # Very large values
#         x_large = np.array([1e10, 1e20, 1e30])
#         result_large = self.fitter._generalized_polynomial(x_large, 1, 1, 2)
#         # Assert that result_large is finite
#         assert np.all(np.isfinite(result_large)), "result_large is not a finite value"
#         # Very small values
#         x_small = np.array([1e-10, 1e-20, 1e-30])
#         result_small = self.fitter._generalized_polynomial(x_small, 1, 1, 2)
#         # Assert that result_small is finite
#         assert np.all(np.isfinite(result_small)), "result_large is not a finite value"
#         # Extreme parameters
#         x = 2
#         result_extreme = self.fitter._generalized_polynomial(x, 1e100, 1e-100, 50)
#         # Assert that result_extreme is finite
#         assert np.all(np.isfinite(result_extreme)), "result_large is not a finite value"
        

#     def test_different_input_types(self):
#         """
#         GIVEN: Different input types
#         WHEN: The polynomial scaling function is applied
#         THEN: The function works consistently across input types
#         """
#         x_scalar = 2
#         x_list = [1, 2, 3]
#         x_numpy = np.array([1, 2, 3])
        
#         result_scalar = self.fitter._generalized_polynomial(x_scalar, 1, 2, 5)
#         with pytest.warns(UserWarning):
#             result_list = self.fitter._generalized_polynomial(x_list, 1, 2, 4.5)
#         result_numpy = self.fitter._generalized_polynomial(x_numpy, 1, 2, -2)
        
#         # Validate results are consistent
#         assert isinstance(result_scalar, (int, float, np.generic)), "result_scalar must be an int, float, or a NumPy scalar"
#         assert len(result_list) == 3
#         assert len(result_numpy) == 3
        
#     def test_invalid_exponent_type(self):
#         """ Test invalid exponent type 
#         GIVEN: Different invalid input types for the exponent r 
#         WHEN: The polynomial scaling function is applied
#         THEN: A ValueError is raised 
#         """

#         x_numpy = np.array([1, 2, 3])
#         # Test with various invalid types
#         invalid_types = [
#             "string", 
#             [1, 2], 
#             {"key": "value"}, 
#             None, 
#             complex(1, 1)
#         ]
        
#         # Check that ValueError is raised for each invalid type
#         for invalid_type in invalid_types:
#             with pytest.raises(ValueError, match="Exponent r should be an integer or float."):
#                 self.fitter._generalized_polynomial(x_numpy, 1, 2, invalid_type)
#     # Power law test
#     @given(
#         a=st.floats(min_value=-10, max_value=10),
#         b=st.floats(min_value=-10, max_value=10),
#         r=one_of(
#         st.floats(min_value=-5, max_value=0),  # negative values up to 0
#         st.floats(min_value=4, max_value=10)   # values from 4 to 10
#     )
# )
#     def test_power_law_fit(self, fitter, data_edge_cases, a, b, r):
#         x = data_edge_cases
#         y = fitter._generalized_polynomial(x, a, b, r)
#         fitting_results = fitter.calculate_non_linear_fit(x,y)
#         # Find the results for combination_of_exponential
#         combination_exp_result = None
#         for result in fitting_results:
#             if result['function'] == 'combination_of_exponential':
#                 combination_exp_result = result
#                 break
        
#         # Assert that we found the combination_of_exponential results
#         assert combination_exp_result is not None, (
#             "combination_of_exponential results not found in fitting results"
#         )
        
#         # Check success status
#         assert combination_exp_result['success'] == True, (
#             "Fitting with combination_of_exponential was not successful"
#         )
#         # best_funct, coeffs, score, _ = self._get_best_fit(fitter, x, y)
#         # assert best_funct == 'generalized_polynomial'
        # assert score < 1e-3
# @pytest.mark.parametrize("num_samples", [
#     50,    # Small number of samples
#     100,   # Moderate number of samples
#     5000,  # Default value
#     10000  # Large number of samples
# ])
# def test_varying_sample_sizes(self, fitter, test_data, num_samples, tolerance = 0.1):
#     """
#     Test behavior with different numbers of samples.
#     """
#     # Create some reasonable test data
#     true_params = [2, -0.3, 1]
#     y = fitter._exponential(test_data, *true_params)
    
#     # Get Bayesian guess with specified number of samples
#     bayesian_guess = fitter._generate_bayesian_initial_guess(
#         fitter._exponential,
#         test_data,
#         y,
#         num_samples=num_samples
#     )
    
#     # Verify that the estimated parameters are close to the true parameters
#     for true_param, estimated_param in zip(true_params, bayesian_guess):
#         assert abs(true_param - estimated_param) < tolerance, \
#             f"Estimated parameter {estimated_param} is not close to true parameter {true_param} (tolerance={tolerance})"
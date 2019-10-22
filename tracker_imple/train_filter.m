function [ctheta_f, theta_pre_f, L] = train_filter(ctheta_f, xlf, yf, ...
    theta_pre_f, params, output_sz, seq)

    for k = 1: numel(xlf)
        x_f = xlf{k};

        if (seq.frame == 1)
            theta_pre_f{k} = zeros(size(x_f));
            lambda_2 = 0;
        else
            lambda_2 = params.lambda2(k);
        end
        
        % intialisation
        theta_f = single(zeros(size(x_f)));
        theta_prime_f = theta_f;
        eta_f = theta_f;
        mu  = params.init_penalty_factor(k);
        mu_scale_step = params.penalty_scale_step(k);

        % pre-compute the variables
        T = prod(output_sz);
        S_xx = sum(conj(x_f) .* x_f, 3);
        Stheta_pre_f = sum(conj(x_f) .* theta_pre_f{k}, 3);
        Sfx_pre_f = bsxfun(@times, x_f, Stheta_pre_f);

        % solve via ADMM algorithm
        iter = 1;
        while (iter <= params.max_iterations)

            % solving theta
            B = S_xx + T * (mu + lambda_2);
            Sgx_f = sum(conj(x_f) .* theta_prime_f, 3);
            Shx_f = sum(conj(x_f) .* eta_f, 3);
 
            theta_f = ((1/(T*(mu + lambda_2)) * bsxfun(@times,  yf{k}, x_f)) ...
                - ((1/(mu + lambda_2)) * eta_f) +(mu/(mu + lambda_2)) ...
                * theta_prime_f) + (lambda_2/(mu + lambda_2)) * theta_pre_f{k} ...
                - bsxfun(@rdivide,(1/(T*(mu + lambda_2)) * bsxfun(@times, ...
                x_f, (S_xx .*  yf{k})) + (lambda_2/(mu + lambda_2)) * Sfx_pre_f - ...
                (1/(mu + lambda_2))* (bsxfun(@times, x_f, Shx_f)) +(mu/(mu ...
                + lambda_2))* (bsxfun(@times, x_f, Sgx_f))), B);

            % solving theta_prime
            X = real(ifft2(mu * theta_f+ eta_f));
            if (seq.frame == 1)
                X_temp = zeros(size(X));     
                for i = 1:size(X,3)
                 X_temp(:,:,i) = X(:,:,i) ./  (params.reg_window{k} .^2 + mu);
                end
                L = 0;
            else  
            X_temp=X;
            L{k} = max(0,1-1./(mu*numel(X)*sqrt(sum(X_temp.^2,3))));
    
            [~,b] = sort(L{k}(:),'descend');
            L{k}(b(ceil(params.fs_rate(k)*1/params.search_area_scale^2*numel(b)):end)) = 0;
    
            X_temp = repmat(L{k},1,1,size(X_temp,3)) .* X_temp;
            end
            
            theta_prime_f = fft2(X_temp);

            %   update eta
            eta_f = eta_f + (mu * (theta_f - theta_prime_f));

            %   update mu
            mu = min(mu_scale_step * mu, 0.1);
            
            iter = iter+1;
        end
        
        % save the trained filters
        theta_pre_f{k} = theta_f;
        
        if seq.frame == 1
            ctheta_f{k} = theta_f;
        else
            ctheta_f{k} = params.rl * theta_f + (1-params.rl) * ctheta_f{k};
        end
    end  
    
end


# Hidden Markov Model classifier.
# Currently capable of:
#	-supervised training on data with arbitrary state and observation domains.
# 	-decoding of obsservation strings via Viterbi
#	-computing token level accuracy across a list of observation sequences
#		against a provided gold standard

require 'rubygems'
require 'narray'

class Array; def sum; inject( nil ) { |sum,x| sum ? sum+x : x }; end; end

class HMM
	
	Infinity = 1.0/0

	class Classifier
		attr_accessor :a, :b, :pi, :o_lex, :q_lex, :debug, :train
		# Member variables:
		# 	pi -- initial state distribution
		#	a -- state transition probabilities
		#	b -- state-conditional observation probabilities
		#	o_lex -- index of observation labels
		#	q_lex -- index of state labels
		#	debug -- flag for verbose output to stdout
		#	train -- a list of labelled sequences for supervised training
		
		def initialize
			@o_lex, @q_lex, @train = [], [], []
		end
		
		def add_to_train(o, q)
			@o_lex |= o # add new tokens to indexed lexicon
			@q_lex |= q
			@train << Sequence.new(index(o, @o_lex), index(q, @q_lex))
		end
		
		def train
			# initialize Pi, A, and B
			@pi = NArray.float(@q_lex.length)
			@a = NArray.float(@q_lex.length, @q_lex.length)
			@b = NArray.float(@q_lex.length, @o_lex.length)
			
			# count frequencies to build Pi, A, and B
			@train.each do |sequence|
				@pi[sequence.q.first] +=1
				sequence.q.length.times do |i|
					@b[sequence.q[i], sequence.o[i]] += 1
					@a[sequence.q[i-1], sequence.q[i]] +=1 if i>0
				end
			end
			
			# smooth to allow unobserved cases
			@pi += 0.1
			@a += 0.1
			@b += 0.1
			
			# normalize frequencies into probabilities
			@pi /= @pi.sum
			@a /= @a.sum(1)
			@b /= @b.sum(1)
		end	
		
		def train_unsupervised2(sequences)
			# for debugging ONLY
			orig_sequences = sequences.clone
			sequences = [sequences.sum]
			
			# initialize model parameters if we don't already have an estimate
			@pi ||= NArray.float(@q_lex.length).fill(1)/@q_lex.length			
			@a ||= NArray.float(@q_lex.length, @q_lex.length).fill(1)/@q_lex.length
			@b ||= NArray.float(@q_lex.length, @o_lex.length).fill(1)/@q_lex.length
			puts @pi.inspect, @a.inspect, @b.inspect if debug
			
			max_iterations = 1 #1000 #kwargs.get('max_iterations', 1000)
			epsilon = 1e-6 # kwargs.get('convergence_logprob', 1e-6)
			
			max_iterations.times do |iteration|
				puts "iteration ##{iteration}" #if debug
				logprob = 0.0
				
				sequences.each do |sequence|
					# just in case, skip if sequence contains unrecognized tokens
					next unless (sequence-o_lex).empty?
					
					# compute forward and backward probabilities
					alpha = forward_probability(sequence)
					beta = backward_probability(sequence)
					lpk = log_add(alpha[-1, true]) #sum of last alphas. divide by this to get probs
					logprob += lpk
					
					xi = xi(sequence)
					gamma = gamma(xi)
					
					localA = NArray.float(q_lex.length,q_lex.length)
					localB = NArray.float(q_lex.length,o_lex.length)
					
					q_lex.each_index do |i|
						q_lex.each_index do |j|
							numA = -Infinity
							denomA = -Infinity
							sequence.each_index do |t|
								break if t >= sequence.length-1
								numA = log_add([numA, xi[t, i, j]])
								denomA = log_add([denomA, gamma[t, i]])
							end
							localA[i,j] = numA - denomA
						end
						
						o_lex.each_index do |k|
							numB = -Infinity
							denomB = -Infinity
							sequence.each_index do |t|
								break if t >= sequence.length-1
								denomB = log_add([denomB, gamma[t, i]])
								next unless k == index(sequence[t], o_lex)
								numB = log_add([numB, gamma[t, i]])
							end
							localB[i, k] = numB - denomB
						end
						
					end
					
					puts "LogProb: #{logprob}"
					
					@a = localA.collect{|x| Math::E**x}
					@b = localB.collect{|x| Math::E**x}
					#@pi = gamma[0, true] / gamma[0, true].sum
					
				end
			end
		end
		
		
		def train_unsupervised(sequences, max_iterations = 10)
			# initialize model parameters if we don't already have an estimate
			@pi ||= NArray.float(@q_lex.length).fill(1)/@q_lex.length			
			@a ||= NArray.float(@q_lex.length, @q_lex.length).fill(1)/@q_lex.length
			@b ||= NArray.float(@q_lex.length, @o_lex.length).fill(1)/@q_lex.length
			puts @pi.inspect, @a.inspect, @b.inspect if debug
			
			converged = false
			last_logprob = 0
			iteration = 0
			#max_iterations = 10 #1000 #kwargs.get('max_iterations', 1000)
			epsilon = 1e-6 # kwargs.get('convergence_logprob', 1e-6)
			
			max_iterations.times do |iteration|
				puts "iteration ##{iteration}" #if debug

				_A_numer = NArray.float(q_lex.length,q_lex.length).fill(-Infinity)
				_B_numer = NArray.float(q_lex.length, o_lex.length).fill(-Infinity)
				_A_denom = NArray.float(q_lex.length).fill(-Infinity)
				_B_denom = NArray.float(q_lex.length).fill(-Infinity)
				_Pi = NArray.float(q_lex.length)
				
				logprob = 0.0
				
				#logprob = last_logprob + 1 # take this out
				
				sequences.each do |sequence|
					# just in case, skip if sequence contains unrecognized tokens
					next unless (sequence-o_lex).empty?
					
					# compute forward and backward probabilities
					alpha = forward_probability(sequence)
					beta = backward_probability(sequence)
					lpk = log_add(alpha[-1, true]) #sum of last alphas. divide by this to get probs
					logprob += lpk
					
					local_A_numer = NArray.float(q_lex.length,q_lex.length).fill(-Infinity)
					local_B_numer = NArray.float(q_lex.length, o_lex.length).fill(-Infinity)
					local_A_denom = NArray.float(q_lex.length).fill(-Infinity)
					local_B_denom = NArray.float(q_lex.length).fill(-Infinity)
					local_Pi = NArray.float(q_lex.length)
					
					sequence.each_with_index do |o, t|
						o_next = index(sequence[t+1], o_lex) if t < sequence.length-1
						
						q_lex.each_index do |i|
							if t < sequence.length-1
								q_lex.each_index do |j|
									local_A_numer[i, j] =  \
										log_add([local_A_numer[i, j], \
										alpha[t, i] + \
											log(@a[i,j]) + \
											log(@b[j,o_next]) + \
											beta[t+1, j]])
								end
								local_A_denom[i] = log_add([local_A_denom[i],
											alpha[t, i] + beta[t, i]])
	
							else
								local_B_denom[i] = log_add([local_A_denom[i],
											alpha[t, i] + beta[t, i]])
							end
							local_B_numer[i, index(o,o_lex)] = log_add([local_B_numer[i, index(o, o_lex)],
								alpha[t, i] + beta[t, i]])

						end
						
						puts local_A_numer.inspect if debug
						
						q_lex.each_index do |i|
							q_lex.each_index do |j|
								_A_numer[i, j] = log_add([_A_numer[i, j],
									local_A_numer[i, j] - lpk])
							end
							o_lex.each_index do |k|	
								_B_numer[i, k] = log_add([_B_numer[i, k], local_B_numer[i, k] - lpk])
							end
							_A_denom[i] = log_add([_A_denom[i], local_A_denom[i] - lpk])
							_B_denom[i] = log_add([_B_denom[i], local_B_denom[i] - lpk])
						end
						
					end
				
					puts alpha.collect{|x| Math::E**x}.inspect if debug
				end		

				puts _A_denom.inspect if debug

				q_lex.each_index do |i|
					q_lex.each_index do |j|
						#puts 2**(_A_numer[i,j] - _A_denom[i]), _A_numer[i,j], _A_denom[i]
						@a[i, j] = Math::E**(_A_numer[i,j] - _A_denom[i])
					end
					o_lex.each_index do |k|
						@b[i, k] = Math::E**(_B_numer[i,k] - _B_denom[i])
					end
					# This comment appears in NLTK:
					# Rabiner says the priors don't need to be updated. I don't
					# believe him. FIXME
				end
					

				if iteration > 0 and (logprob - last_logprob).abs < epsilon
					puts "CONVERGED: #{(logprob - last_logprob).abs}" if debug
					puts "epsilon: #{epsilon}" if debug
					break
				end
				
				puts "LogProb: #{logprob}" #if debug
				
				last_logprob = logprob
			end
		end
		
		def xi(sequence)
			xi = NArray.float(sequence.length-1, q_lex.length, q_lex.length)
			
			alpha = forward_probability(sequence)
			beta = backward_probability(sequence)
			
			0.upto sequence.length-2 do |t|
				denom = 0
				q_lex.each_index do |i|
					q_lex.each_index do |j|
						x = alpha[t, i] + log(@a[i,j]) + \
							log(@b[j,index(sequence[t+1], o_lex)]) + \
							beta[t+1, j]
						denom = log_add([denom, x])
					end
				end
				
				q_lex.each_index do |i|
					q_lex.each_index do |j|
						numer = alpha[t, i] + log(@a[i,j]) + \
							log(@b[j,index(sequence[t+1], o_lex)]) + \
							beta[t+1, j]
						xi[t, i, j] = numer - denom
					end
				end
			end
			
			puts "Xi: #{xi.inspect}" if debug
			xi
		end
		
		def gamma(xi)
			gamma = NArray.float(xi.shape[0], xi.shape[1]).fill(-Infinity)
			
			0.upto gamma.shape[0] - 1 do |t|
				q_lex.each_index do |i|
					q_lex.each_index do |j|
						gamma[t, i] = log_add([gamma[t, i], xi[t, i, j]])
					end
				end
			end
			
			puts "Gamma: #{gamma.inspect}" if debug
			gamma
		end
		
		def forward_probability(sequence)
			alpha = NArray.float(sequence.length, q_lex.length).fill(-Infinity)
			
			alpha[0, true] = log(@pi) + log(@b[true, index(sequence.first, o_lex)])
			
			sequence.each_with_index do |o, t|
				next if t==0
				q_lex.each_index do |i|
					q_lex.each_index do |j|
						alpha[t, i] = log_add([alpha[t, i], alpha[t-1, j]+log(@a[j, i])])
					end
					alpha[t, i] += log(b[i, index(o, o_lex)])
				end
			end
			alpha
		end
		
		def log_add(values)
			x = values.max
			if x > -Infinity
				sum_diffs = 0
				values.each do |value|
					sum_diffs += Math::E**(value - x)
				end
				return x + log(sum_diffs)
			else
				return x
			end
		end
		
		def backward_probability(sequence)
			beta = NArray.float(sequence.length, q_lex.length).fill(-Infinity)
			
			beta[-1, true] = log(1)
			
			(sequence.length-2).downto(0) do |t|
				q_lex.each_index do |i|
					q_lex.each_index do |j|
						beta[t, i] = log_add([beta[t,i], log(@a[i, j]) \
							+ log(@b[j, index(sequence[t+1], o_lex)]) \
							+ beta[t+1, j]])
					end
				end
			end

			beta
		end
		
		def decode(o_sequence)
			# Viterbi!  with log probability math to avoid underflow
			
			# encode observations
			o_sequence = index(o_sequence, @o_lex)
			
			# initialize.  skipping the 0 initialization for psi, as it's never used.
			# psi will have T-1 elements instead of T, allowing it
			# to control the backtrack iterator later.
			delta, psi = [log(pi)+log(b[true, o_sequence.shift])], []

			# recursive step
			o_sequence.each do |o|
				psi << argmax(delta.last+log(a))
				delta << (delta.last+log(a)).max(0)+log(b[true, o])
			end
			
			# initialize Q* with final state
			q_star = [delta.last.sort_index[-1]]
			
			# backtrack the optimal state sequence into Q*
			psi.reverse.each do |psi_t|
				q_star.unshift psi_t[q_star.first]
			end
			
			puts "delta:", exp(delta).inspect, "psi:", exp(psi).inspect if @debug
			
			return deindex(q_star, @q_lex)
		end
		
		def accuracy(o, q)
			# token level accuracy across a set of sequences
			correct, total = 0.0, 0.0
			o.length.times do |i|
				correct += (NArray.to_na(decode(o[i])).eq NArray.to_na(q[i])).sum
				total += o[i].length
			end
			correct/total
		end
		
		private
		
		# index and deindex map between labels and the ordinals of those labels.
		# the ordinals map the labels to rows and columns of Pi, A, and B
		def index(subject, lexicon)
			if subject.is_a?(Array) or subject.is_a?(NArray)
				return subject.collect{|x| lexicon.rindex(x)}
			else
				return index(Array[subject], lexicon)[0]
			end
		end
		
		#private
		
		def deindex(sequence, lexicon)
			sequence.collect{|i| lexicon[i]}
		end
		
		# abstracting out some array element operations for readability
		def log(subject)
			if subject.is_a?(Array) or subject.is_a?(NArray)
				return subject.collect{|n| NMath::log n}
			else
				return log(Array[subject])[0]
			end
		end
		
		def exp(array)
			# e to the power of each element
			array.collect{|n| Math::E ** n}
		end
		
		def argmax(narray)
			# horizontal index of the max in each row.
			# the mod is b/c sort_index returns global indices
			# (rather than starting at 0 for each row)
			(narray).sort_index(0)[-1, true] % narray.shape[1]
		end
	end

	class Sequence
	      attr_accessor :o, :q # array of observations, array of states
	      def initialize (o, q)
	      	  @o, @q = o, q
	      end
	end
end
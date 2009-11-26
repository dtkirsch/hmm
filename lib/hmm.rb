# Hidden Markov Model classifier.
# Currently capable of:
#	-supervised training on data with arbitrary state and observation domains.
# 	-decoding of obsservation strings via Viterbi
#	-computing token level accuracy across a list of observation sequences
#		against a provided gold standard


require 'rubygems'
require 'narray'

class HMM

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
			
			# normalize frequencies into probabilities
			@pi /= @pi.sum
			@a /= @a.sum(1)
			@b /= @b.sum(1)
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
		def index(sequence, lexicon)
                        lexicon |= sequence # add any unknown tokens to the lex
			sequence.collect{|x| lexicon.rindex(x)}
		end
		
		def deindex(sequence, lexicon)
			sequence.collect{|i| lexicon[i]}
		end
		
		# abstracting out some array element operations for readability
		def log(array)
			# natural log of each element
			array.collect{|n| NMath::log n}
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
